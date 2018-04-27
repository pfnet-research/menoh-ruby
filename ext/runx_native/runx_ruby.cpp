#include <iostream>
#include <new>

#include "runx_ruby.hpp"

struct runx_ruby {
    runx::model_data* onnx;
};

static runx_ruby* getONNX(VALUE self) {
    runx_ruby* p;
    Data_Get_Struct(self, runx_ruby, p);
    return p;
}

static void wrap_runx_free(runx_ruby* p) {
    delete p->onnx;
    ruby_xfree(p);
}

static VALUE wrap_runx_alloc(VALUE klass) {
    void* p = ruby_xmalloc(sizeof(runx_ruby));
    return Data_Wrap_Struct(klass, NULL, wrap_runx_free, p);
}

static VALUE wrap_runx_init(VALUE self, VALUE vfilename) {
    char* filename = StringValuePtr(vfilename);
    // Load ONNX model
    getONNX(self)->onnx = new runx::model_data(runx::load_onnx(filename));

    return Qnil;
}

struct runxModel {
    runx::model_data* onnx;
    runx::model* model;
    std::string* backend;
    std::vector<std::string>* output_layers;
};

static runxModel* getModel(VALUE self) {
    runxModel* p;
    Data_Get_Struct(self, runxModel, p);
    return p;
}

static void wrap_model_free(runxModel* p) {
    delete p->model;
    delete p->backend;
    delete p->output_layers;
    ruby_xfree(p);
}

static VALUE wrap_model_alloc(VALUE klass) {
    void* p = ruby_xmalloc(sizeof(runxModel));
    return Data_Wrap_Struct(klass, NULL, wrap_model_free, p);
}

static VALUE wrap_model_init(VALUE self, VALUE vonnx, VALUE condition) {

    // condition
    getModel(self)->onnx = getONNX(vonnx)->onnx;
    VALUE vbackend =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("backend")));
    getModel(self)->backend = new std::string(StringValuePtr(vbackend));

    // output_layer
    std::vector<std::string>* output_layers = new std::vector<std::string>;
    VALUE voutput_layers =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("output_layers")));
    int output_layer_num =
      NUM2INT(rb_funcall(voutput_layers, rb_intern("length"), 0, NULL));
    for(int i = 0; i < output_layer_num; i++) {
        VALUE voutput_layer = rb_ary_entry(voutput_layers, i);
        output_layers->push_back(std::string(StringValuePtr(voutput_layer)));
    }
    getModel(self)->output_layers = output_layers;

    return Qnil;
}

static VALUE wrap_model_run(VALUE self, VALUE dataset, VALUE condition) {

    // condition
    int channel_num = NUM2INT(
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("channel_num"))));
    int height =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("height"))));
    int width =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("width"))));

    // input_layer
    VALUE vinput_layer =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("input_layer")));
    std::string* input_layer = new std::string(StringValuePtr(vinput_layer));

    int batch_size = NUM2INT(rb_funcall(dataset, rb_intern("length"), 0, NULL));
    int array_length = channel_num * width * height;

    std::vector<int> input_dims{batch_size, channel_num, height, width};

    runx::model* model = new runx::model(
        {std::make_pair(*input_layer, input_dims)},
        *(getModel(self)->output_layers),
        *(getModel(self)->onnx),
        *(getModel(self)->backend)
    );
    getModel(self)->model = model;

    // Flatten and cast Array for Runx
    std::vector<float> image_data(batch_size * array_length);
    for(int i = 0; i < batch_size; i++) {
        VALUE data = rb_ary_entry(dataset, i);
        int data_offset = i * array_length;
        for(int j = 0; j < array_length; ++j) {
            image_data[data_offset + j] = static_cast<float>(NUM2DBL(rb_ary_entry(data, j)));
        }
    }

    // Copy input image data to model's input array
    auto& input_array =
      model->input(*input_layer);
    std::copy(image_data.begin(), image_data.end(),
              runx::fbegin(input_array));

    // Run inference
    model->run();

    // Get output
    VALUE results = rb_ary_new();
    for(int i = 0; i < batch_size; i++) {
        VALUE result_each = rb_hash_new();
        for(auto output_layer : *(getModel(self)->output_layers)) {
            auto const& out_arr = model->output(output_layer);
            int unit_num =
              runx::total_size(out_arr) / batch_size;
            // Convert result to Ruby Array
            VALUE result_layer_output = rb_ary_new();
            for(int j = i * unit_num; j < (i + 1) * unit_num; ++j) {
                rb_ary_push(result_layer_output,
                            DBL2NUM(runx::fat(out_arr, j)));
            }

            rb_hash_aset(result_each, rb_str_new2(output_layer.c_str()),
                         result_layer_output);
        }
        rb_ary_push(results, result_each);
    }

    return results;
}

/**
 * will be called when required
 */

VALUE mRunx;

extern "C" void Init_runx_native() {

    mRunx = rb_define_module("Runx");

    VALUE onnx = rb_define_class_under(mRunx, "Runx", rb_cObject);

    rb_define_alloc_func(onnx, wrap_runx_alloc);
    rb_define_private_method(onnx, "native_init",
                             RUBY_METHOD_FUNC(wrap_runx_init), 1);

    VALUE model = rb_define_class_under(mRunx, "RunxModel", rb_cObject);

    rb_define_alloc_func(model, wrap_model_alloc);
    rb_define_private_method(model, "native_init",
                             RUBY_METHOD_FUNC(wrap_model_init), 2);

    rb_define_private_method(model, "native_run", RUBY_METHOD_FUNC(wrap_model_run), 2);
}
