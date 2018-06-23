#include "runx_ruby.h"

#define ERROR_CHECK(statement, exceptiontype)                                  \
  {                                                                            \
    menoh_error_code ec = statement;                                           \
    if (ec) {                                                                  \
      rb_raise(exceptiontype, "%s", menoh_get_last_error_message());           \
      return Qnil;                                                             \
    }                                                                          \
  }

typedef struct runx_ruby {
  menoh_model_data_handle model_data;
} runx_ruby;

static runx_ruby *getONNX(VALUE self) {
  runx_ruby *p;
  Data_Get_Struct(self, runx_ruby, p);
  return p;
}

static void wrap_runx_free(runx_ruby *p) {
  if (p) {
    menoh_delete_model_data(p->model_data); // TODO
    ruby_xfree(p);
  }
}

static VALUE wrap_runx_alloc(VALUE klass) {
  void *p = ruby_xmalloc(sizeof(runx_ruby));
  memset(p, 0, sizeof(runx_ruby));
  return Data_Wrap_Struct(klass, NULL, wrap_runx_free, p);
}

static VALUE wrap_runx_init(VALUE self, VALUE vfilename) {
  menoh_error_code ec = menoh_error_code_success;
  char *filename = StringValuePtr(vfilename);
  // Load ONNX model

  menoh_model_data_handle model_data;
  ERROR_CHECK(menoh_make_model_data_from_onnx(filename, &model_data),
              rb_eArgError);
  getONNX(self)->model_data = model_data;

  return Qnil;
}

typedef struct runxModel {
  menoh_model_data_handle model_data;
  VALUE vbackend;
  VALUE voutput_layers;
} runxModel;

static runxModel *getModel(VALUE self) {
  runxModel *p;
  Data_Get_Struct(self, runxModel, p);
  return p;
}

static void wrap_model_free(runxModel *p) {
  if (p) {
    ruby_xfree(p);
  }
}

static VALUE wrap_model_alloc(VALUE klass) {
  void *p = ruby_xmalloc(sizeof(runxModel));
  memset(p, 0, sizeof(runxModel));
  return Data_Wrap_Struct(klass, NULL, wrap_model_free, p);
}

static VALUE wrap_model_init(VALUE self, VALUE vonnx, VALUE condition) {

  // condition
  getModel(self)->model_data = getONNX(vonnx)->model_data;
  VALUE vbackend =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("backend")));
  getModel(self)->vbackend = vbackend;

  // output_layer
  VALUE voutput_layers =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("output_layers")));
  getModel(self)->voutput_layers = voutput_layers;

  return Qnil;
}

static VALUE wrap_model_prepare(VALUE self, VALUE vbatchsize, VALUE condition) {
}

static VALUE wrap_model_run(VALUE self, VALUE dataset, VALUE condition) {

  // condition
  int32_t channel_num = NUM2INT(
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("channel_num"))));
  int32_t height =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("height"))));
  int32_t width =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("width"))));

  int32_t batch_size =
      NUM2INT(rb_funcall(dataset, rb_intern("length"), 0, NULL));
  int32_t array_length = channel_num * width * height;

  VALUE vbackend = getModel(self)->vbackend;
  VALUE vinput_layer =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("input_layer")));
  VALUE voutput_layers = getModel(self)->voutput_layers;

  //////////////////////////
  // get vpt builder
  menoh_variable_profile_table_builder_handle vpt_builder;
  ERROR_CHECK(menoh_make_variable_profile_table_builder(&vpt_builder),
              rb_eStandardError);

  // set output_layer
  int32_t output_layer_num = NUM2INT(
      rb_funcall(getModel(self)->voutput_layers, rb_intern("length"), 0, NULL));
  for (int32_t i = 0; i < output_layer_num; i++) {
    VALUE voutput_layer = rb_ary_entry(getModel(self)->voutput_layers, i);
    ERROR_CHECK(
        menoh_variable_profile_table_builder_add_output_profile(
            vpt_builder, StringValuePtr(voutput_layer), menoh_dtype_float),
        rb_eStandardError);
  }

  // set input layer
  ERROR_CHECK(menoh_variable_profile_table_builder_add_input_profile_dims_4(
                  vpt_builder, StringValuePtr(vinput_layer), menoh_dtype_float,
                  batch_size, channel_num, width, height),
              rb_eStandardError);

  // build variable provile table
  menoh_variable_profile_table_handle variable_profile_table;
  ERROR_CHECK(menoh_build_variable_profile_table(vpt_builder,
                                                 getModel(self)->model_data,
                                                 &variable_profile_table),
              rb_eStandardError);

  // optimize
  ERROR_CHECK(menoh_model_data_optimize(getModel(self)->model_data,
                                        variable_profile_table),
              rb_eStandardError);

  // get model buildler
  menoh_model_builder_handle model_builder;
  ERROR_CHECK(menoh_make_model_builder(variable_profile_table, &model_builder),
              rb_eStandardError);

  // attach input buffer to model builder
  float *input_buff;
  input_buff = (float *)malloc(sizeof(float) * batch_size * channel_num *
                               width * height);
  ERROR_CHECK(menoh_model_builder_attach_external_buffer(
                  model_builder, StringValuePtr(vinput_layer), input_buff),
              rb_eStandardError);

  // build model
  menoh_model_handle model;
  ERROR_CHECK(menoh_build_model(model_builder, getModel(self)->model_data,
                                StringValuePtr(vbackend), "", &model),
              rb_eStandardError);

  // Copy input image data to model's input array
  for (int32_t i = 0; i < batch_size; i++) {
    VALUE data = rb_ary_entry(dataset, i);
    int32_t data_offset = i * array_length;
    for (int32_t j = 0; j < array_length; ++j) {
      input_buff[data_offset + j] = (float)(NUM2DBL(rb_ary_entry(data, j)));
    }
  }

  // attach output buffer to model
  float **output_buffs;
  output_buffs = (float **)malloc(sizeof(float *) * output_layer_num);
  for (int32_t i = 0; i < output_layer_num; i++) {
    VALUE voutput_layer = rb_ary_entry(getModel(self)->voutput_layers, i);
    float *output_buff;
    ERROR_CHECK(
        menoh_model_get_variable_buffer_handle(
            model, StringValuePtr(voutput_layer), (void **)&output_buff),
        rb_eStandardError);
    output_buffs[i] = output_buff;
  }

  // run model
  ERROR_CHECK(menoh_model_run(model), rb_eStandardError);

  // Get output
  VALUE results = rb_ary_new();
  for (int32_t output_layer_i = 0; output_layer_i < output_layer_num;
       output_layer_i++) {
    VALUE voutput_layer =
        rb_ary_entry(getModel(self)->voutput_layers, output_layer_i);
    VALUE result_each = rb_hash_new();

    // get dimention of output layers
    int32_t dim_size;
    int32_t output_buffer_length = 1;
    ERROR_CHECK(
        menoh_variable_profile_table_get_dims_size(
            variable_profile_table, StringValuePtr(voutput_layer), &(dim_size)),
        rb_eStandardError);
    VALUE vresult_shape = rb_ary_new();
    // get each size of dimention
    for (int32_t dim = 0; dim < dim_size; dim++) {
      int32_t size;
      ERROR_CHECK(menoh_variable_profile_table_get_dims_at(
                      variable_profile_table, StringValuePtr(voutput_layer),
                      dim, &(size)),
                  rb_eStandardError);
      rb_ary_push(vresult_shape, INT2NUM(size));
      output_buffer_length *= size;
    }

    // Convert result to Ruby Array
    VALUE vresult_buffer = rb_ary_new();
    for (int32_t j = 0; j < output_buffer_length; j++) {
      float *output_buff;
      output_buff = output_buffs[output_layer_i];
      rb_ary_push(vresult_buffer, DBL2NUM(*(output_buff + j)));
    }

    rb_hash_aset(result_each, rb_to_symbol(rb_str_new2("name")), voutput_layer);
    rb_hash_aset(result_each, rb_to_symbol(rb_str_new2("shape")),
                 vresult_shape);
    rb_hash_aset(result_each, rb_to_symbol(rb_str_new2("buffer")),
                 vresult_buffer);
    rb_ary_push(results, result_each);
  }

  // TODO error

  menoh_delete_model(model);
  menoh_delete_model_builder(model_builder);
  menoh_delete_variable_profile_table_builder(vpt_builder);

  free(input_buff);
  free(output_buffs);

  return results;
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

  rb_define_private_method(model, "native_run",
                           RUBY_METHOD_FUNC(wrap_model_run), 2);
}
