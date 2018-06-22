#include "runx_ruby.h"

#define ERROR_CHECK(statement)                            \
    {                                                     \
        menoh_error_code ec = statement;                  \
        if(ec) {                                          \
            printf("%s", menoh_get_last_error_message()); \
            return 0;                                     \
        }                                                 \
    }
// TODO return Ruby Error code

typedef struct runx_ruby {
    menoh_model_data_handle model_data;
} runx_ruby;

static runx_ruby* getONNX(VALUE self) {
    runx_ruby* p;
    Data_Get_Struct(self, runx_ruby, p);
    return p;
}

static void wrap_runx_free(runx_ruby* p) {
    if(p){
        menoh_delete_model_data(p->model_data); // TODO
        ruby_xfree(p);
    }
}

static VALUE wrap_runx_alloc(VALUE klass) {
    void* p = ruby_xmalloc(sizeof(runx_ruby));
    memset(p, 0, sizeof(runx_ruby));
    return Data_Wrap_Struct(klass, NULL, wrap_runx_free, p);
}

static VALUE wrap_runx_init(VALUE self, VALUE vfilename) {
    menoh_error_code ec = menoh_error_code_success;
    char* filename = StringValuePtr(vfilename);
    // Load ONNX model

    menoh_model_data_handle model_data;
    ERROR_CHECK(
      menoh_make_model_data_from_onnx(filename, &model_data));
    getONNX(self)->model_data = model_data;

    return Qnil;
}

typedef struct runxModel {
    menoh_model_data_handle model_data;
    VALUE vbackend;
    VALUE voutput_layers;
} runxModel;

static runxModel* getModel(VALUE self) {
    runxModel* p;
    Data_Get_Struct(self, runxModel, p);
    return p;
}

static void wrap_model_free(runxModel* p) {
    if(p){
        ruby_xfree(p);
    }
}

static VALUE wrap_model_alloc(VALUE klass) {
    void* p = ruby_xmalloc(sizeof(runxModel));
    memset(p, 0, sizeof(runxModel));    
    return Data_Wrap_Struct(klass, NULL, wrap_model_free, p);
}

static VALUE wrap_model_init(VALUE self, VALUE vonnx, VALUE condition) {

    // condition
    getModel(self)->model_data = getONNX(vonnx)->model_data;
    VALUE vbackend =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("backend")));
    // getModel(self)->backend = new std::string(StringValuePtr(vbackend));
    getModel(self)->vbackend = vbackend;

    // output_layer
    // std::vector<std::string>* output_layers = new std::vector<std::string>;
    VALUE voutput_layers =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("output_layers")));
    getModel(self)->voutput_layers;

    // for(int i = 0; i < output_layer_num; i++) {
    //     VALUE voutput_layer = rb_ary_entry(voutput_layers, i);
    //     output_layers->push_back(std::string(StringValuePtr(voutput_layer)));
    // }
    // getModel(self)->output_layers = output_layers;

    return Qnil;
}

static VALUE wrap_model_run(VALUE self, VALUE dataset, VALUE condition) {

    // condition
    int32_t channel_num = NUM2INT(
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("channel_num"))));
    int32_t height =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("height"))));
    int32_t width =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("width"))));

    int32_t batch_size = NUM2INT(rb_funcall(dataset, rb_intern("length"), 0, NULL));
    int32_t array_length = channel_num * width * height;

    VALUE vbackend =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("backend")));
    VALUE vinput_layer =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("input_layer")));
    VALUE voutput_layers =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("output_layers")));


//////////////////////////
    // get vpt builder
    menoh_variable_profile_table_builder_handle vpt_builder;
    ERROR_CHECK(menoh_make_variable_profile_table_builder(&vpt_builder));
    ERROR_CHECK(menoh_variable_profile_table_builder_add_input_profile_dims_4(
      vpt_builder, StringValuePtr(vinput_layer), menoh_dtype_float, batch_size, channel_num, width, height));

    menoh_variable_profile_table_handle variable_profile_table;
    ERROR_CHECK(menoh_build_variable_profile_table(vpt_builder, getModel(self)->model_data,
                                                   &variable_profile_table));

    // set output_layer
    int32_t output_layer_num =
      NUM2INT(rb_funcall(getModel(self)->voutput_layers, rb_intern("length"), 0, NULL));
    int32_t* softmax_out_dims;
    softmax_out_dims = (int32_t*)malloc(sizeof(int32_t) * output_layer_num);
    for(int32_t i = 0; i < output_layer_num; i++) {
        VALUE voutput_layer = rb_ary_entry(getModel(self)->voutput_layers, i);
        ERROR_CHECK(menoh_variable_profile_table_builder_add_output_profile(
          vpt_builder, StringValuePtr(voutput_layer), menoh_dtype_float));
        ERROR_CHECK(menoh_variable_profile_table_get_dims_at(
          variable_profile_table, StringValuePtr(voutput_layer), i, &softmax_out_dims[i]));
    }


    ERROR_CHECK(menoh_model_data_optimize(getModel(self)->model_data, variable_profile_table));

    menoh_model_builder_handle model_builder;
    ERROR_CHECK(
      menoh_make_model_builder(variable_profile_table, &model_builder));

    float* input_buff;
    input_buff = (float*)malloc(sizeof(float) * batch_size * channel_num * width * height);
    menoh_model_builder_attach_external_buffer(model_builder, StringValuePtr(vinput_layer),
                                               input_buff);

    menoh_model_handle model;
    ERROR_CHECK(
      menoh_build_model(model_builder, getModel(self)->model_data, StringValuePtr(vbackend), "", &model));


//

    // Copy input image data to model's input array
    float** output_buffs;
    output_buffs = (float **)malloc(sizeof(float) * output_layer_num);
    for(int32_t i = 0; i < output_layer_num; i++) {
        VALUE voutput_layer = rb_ary_entry(getModel(self)->voutput_layers, i);
        ERROR_CHECK(menoh_variable_profile_table_builder_add_output_profile(
          vpt_builder, StringValuePtr(voutput_layer), menoh_dtype_float));
        ERROR_CHECK(menoh_variable_profile_table_get_dims_at(
          variable_profile_table, StringValuePtr(voutput_layer), i, &softmax_out_dims[i]));

        float* output_buff;
        ERROR_CHECK(menoh_model_get_variable_buffer_handle(model, StringValuePtr(voutput_layer),
                                              (void**)&output_buff));
        output_buffs[i] = output_buff;
    }

    // Flatten and cast Array for Runx
    for(int32_t i = 0; i < batch_size; i++) {
        VALUE data = rb_ary_entry(dataset, i);
        int32_t data_offset = i * array_length;
        for(int32_t j = 0; j < array_length; ++j) {
            input_buff[data_offset + j] = (float)(NUM2DBL(rb_ary_entry(data, j)));
        }
    }

    ERROR_CHECK(menoh_model_run(model));

    // for(int i = 0; i < 10; ++i) {
    //     printf("%f ", *(fc6_output_buff + i));
    // }
    // printf("\n");
    // for(int n = 0; n < softmax_out_dims[0]; ++n) {
    //     for(int i = 0; i < softmax_out_dims[1]; ++i) {
    //         printf("%f ", *(softmax_output_buff + n * softmax_out_dims[1] + i));
    //     }
    //     printf("\n");
    // }


//  

    menoh_delete_model(model);
    menoh_delete_model_builder(model_builder);
    menoh_delete_variable_profile_table_builder(vpt_builder);

    free(input_buff);
    free(softmax_out_dims);
    free(output_buffs);

///////////////////////



    // // input_layer
    // VALUE vinput_layer =
    //   rb_hash_aref(condition, rb_to_symbol(rb_str_new2("input_layer")));
    // std::string* input_layer = new std::string(StringValuePtr(vinput_layer)); // TODO change to shared pointer


    // std::vector<int> input_dims{batch_size, channel_num, height, width};

    // try {
    //     menoh::model* model = new menoh::model(
    //         {std::make_pair(*input_layer, input_dims)},
    //         *(getModel(self)->output_layers),
    //         *getModel(self)->model_data,
    //         *(getModel(self)->backend)
    //     );
    //     getModel(self)->model = model;

    //     // Flatten and cast Array for Runx
    //     std::vector<float> image_data(batch_size * array_length);
    //     for(int32_t i = 0; i < batch_size; i++) {
    //         VALUE data = rb_ary_entry(dataset, i);
    //         int32_t data_offset = i * array_length;
    //         for(int32_t j = 0; j < array_length; ++j) {
    //             image_data[data_offset + j] = static_cast<float>(NUM2DBL(rb_ary_entry(data, j)));
    //         }
    //     }

    //     // Copy input image data to model's input array
    //     auto& input_array =
    //     model->input(*input_layer);
    //     std::copy(image_data.begin(), image_data.end(),
    //             menoh::fbegin(input_array));

    //     // Run inference
    //     model->run();

    //     // Get output
    //     VALUE results = rb_ary_new();
    //     for(int32_t i = 0; i < batch_size; i++) {
    //         VALUE result_each = rb_hash_new();
    //         for(auto output_layer : *(getModel(self)->output_layers)) {
    //             auto const& out_arr = model->output(output_layer);
    //             int32_t unit_num =
    //             menoh::total_size(out_arr) / batch_size;
    //             // Convert result to Ruby Array
    //             VALUE result_layer_output = rb_ary_new();
    //             for(int32_t j = i * unit_num; j < (i + 1) * unit_num; ++j) {
    //                 rb_ary_push(result_layer_output,
    //                             DBL2NUM(menoh::fat(out_arr, j)));
    //             }

    //             rb_hash_aset(result_each, rb_str_new2(output_layer.c_str()),
    //                         result_layer_output);
    //         }
    //         rb_ary_push(results, result_each);
    //     }
    //     delete input_layer;
    //     return results;
    // }
    // catch (std::exception& e) {
    //     delete input_layer;
    //     rb_raise(rb_eTypeError, "%s", e.what());
    //     return Qnil;
    // }
}

VALUE mRunx;

void Init_runx_native() {

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
