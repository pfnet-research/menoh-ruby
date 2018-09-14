#include "menoh_ruby.h"

#define ERROR_CHECK(statement, exceptiontype)                        \
  {                                                                  \
    menoh_error_code ec = statement;                                 \
    if (ec) {                                                        \
      rb_raise(exceptiontype, "%s", menoh_get_last_error_message()); \
      return Qnil;                                                   \
    }                                                                \
  }

typedef struct menoh_ruby {
  menoh_model_data_handle model_data;
} menoh_ruby;

static menoh_ruby *getONNX(VALUE self) {
  menoh_ruby *p;
  Data_Get_Struct(self, menoh_ruby, p);
  return p;
}

static void wrap_menoh_free(menoh_ruby *p) {
  if (p) {
    if (p->model_data) menoh_delete_model_data(p->model_data);
    ruby_xfree(p);
  }
}

static VALUE wrap_menoh_alloc(VALUE klass) {
  void *p = ruby_xmalloc(sizeof(menoh_ruby));
  memset(p, 0, sizeof(menoh_ruby));
  return Data_Wrap_Struct(klass, NULL, wrap_menoh_free, p);
}

static VALUE wrap_menoh_init(VALUE self, VALUE vfilename) {
  menoh_error_code ec = menoh_error_code_success;
  FilePathValue(vfilename);
  char *filename = StringValueCStr(vfilename);

  // Load ONNX model
  menoh_model_data_handle model_data;
  ERROR_CHECK(menoh_make_model_data_from_onnx(filename, &model_data),
              rb_eArgError);
  getONNX(self)->model_data = model_data;

  return Qnil;
}

typedef struct menohModel {
  menoh_model_data_handle model_data;
  VALUE vbackend;
  float **input_buffs;
  float **output_buffs;
  menoh_variable_profile_table_builder_handle vpt_builder;
  menoh_variable_profile_table_handle variable_profile_table;
  menoh_model_builder_handle model_builder;
  menoh_model_handle model;
  VALUE vinput_layers;
  VALUE voutput_layers;
  int32_t input_layer_num;
} menohModel;

static menohModel *getModel(VALUE self) {
  menohModel *p;
  Data_Get_Struct(self, menohModel, p);
  return p;
}

static void wrap_model_free(menohModel *p) {
  if (p) {
    if (p->variable_profile_table)
      menoh_delete_variable_profile_table(p->variable_profile_table);
    if (p->vpt_builder)
      menoh_delete_variable_profile_table_builder(p->vpt_builder);
    if (p->model) menoh_delete_model(p->model);
    if (p->model_builder) menoh_delete_model_builder(p->model_builder);
    if (p->input_buffs) {
      for (int32_t i = 0; i < p->input_layer_num; i++) {
        if (p->input_buffs[i]) ruby_xfree(p->input_buffs[i]);
      }
      ruby_xfree(p->input_buffs);
    }
    if (p->output_buffs) ruby_xfree(p->output_buffs);
    ruby_xfree(p);
  }
}

static void wrap_model_mark(menohModel *p) {
  if (p) {
    if (p->vbackend) rb_gc_mark(p->vbackend);
    if (p->vinput_layers) rb_gc_mark(p->vinput_layers);
    if (p->voutput_layers) rb_gc_mark(p->voutput_layers);
  }
}

static VALUE wrap_model_alloc(VALUE klass) {
  void *p = ruby_xmalloc(sizeof(menohModel));
  memset(p, 0, sizeof(menohModel));
  return Data_Wrap_Struct(klass, wrap_model_mark, wrap_model_free, p);
}

static VALUE wrap_model_init(VALUE self, VALUE vonnx, VALUE option) {
  // option
  getModel(self)->model_data = getONNX(vonnx)->model_data;
  VALUE vbackend = rb_hash_aref(option, rb_to_symbol(rb_str_new2("backend")));
  getModel(self)->vbackend = vbackend;

  // option
  VALUE vinput_layers =
      rb_hash_aref(option, rb_to_symbol(rb_str_new2("input_layers")));
  VALUE voutput_layers =
      rb_hash_aref(option, rb_to_symbol(rb_str_new2("output_layers")));

  getModel(self)->vinput_layers = vinput_layers;
  getModel(self)->voutput_layers = voutput_layers;

  // get vpt builder
  ERROR_CHECK(
      menoh_make_variable_profile_table_builder(&(getModel(self)->vpt_builder)),
      rb_eStandardError);

  // set output_layer
  int32_t output_layer_num =
      NUM2INT(rb_funcall(voutput_layers, rb_intern("length"), 0, NULL));
  for (int32_t i = 0; i < output_layer_num; i++) {
    VALUE voutput_layer = rb_ary_entry(voutput_layers, i);
    ERROR_CHECK(menoh_variable_profile_table_builder_add_output_profile(
                    getModel(self)->vpt_builder, StringValueCStr(voutput_layer),
                    menoh_dtype_float),
                rb_eStandardError);
  }

  // set input layer
  int32_t input_layer_num =
      NUM2INT(rb_funcall(vinput_layers, rb_intern("length"), 0, NULL));
  getModel(self)->input_layer_num = input_layer_num;
  for (int32_t i = 0; i < input_layer_num; i++) {
    VALUE vinput_layer = rb_ary_entry(vinput_layers, i);
    VALUE vname = rb_hash_aref(vinput_layer, rb_to_symbol(rb_str_new2("name")));
    VALUE vdims = rb_hash_aref(vinput_layer, rb_to_symbol(rb_str_new2("dims")));
    int32_t dims_length =
        NUM2INT(rb_funcall(vdims, rb_intern("length"), 0, NULL));

    switch (dims_length) {
      case 2:
        ERROR_CHECK(
            menoh_variable_profile_table_builder_add_input_profile_dims_2(
                getModel(self)->vpt_builder, StringValueCStr(vname),
                menoh_dtype_float, NUM2INT(rb_ary_entry(vdims, 0)),
                NUM2INT(rb_ary_entry(vdims, 1))),
            rb_eStandardError);
        break;
      case 4:
        ERROR_CHECK(
            menoh_variable_profile_table_builder_add_input_profile_dims_4(
                getModel(self)->vpt_builder, StringValueCStr(vname),
                menoh_dtype_float, NUM2INT(rb_ary_entry(vdims, 0)),
                NUM2INT(rb_ary_entry(vdims, 1)),
                NUM2INT(rb_ary_entry(vdims, 2)),
                NUM2INT(rb_ary_entry(vdims, 3))),
            rb_eStandardError);
        break;
      default:
        rb_raise(rb_eStandardError, "invalid dimension length");
        return Qnil;
    }
  }

  // build variable provile table
  ERROR_CHECK(menoh_build_variable_profile_table(
                  getModel(self)->vpt_builder, getModel(self)->model_data,
                  &(getModel(self)->variable_profile_table)),
              rb_eStandardError);

  // optimize
  ERROR_CHECK(
      menoh_model_data_optimize(getModel(self)->model_data,
                                getModel(self)->variable_profile_table),
      rb_eStandardError);

  // get model buildler
  ERROR_CHECK(menoh_make_model_builder(getModel(self)->variable_profile_table,
                                        &(getModel(self)->model_builder)),
              rb_eStandardError);

  // attach input buffer to model builder
  getModel(self)->input_buffs =
      (float **)ruby_xmalloc(sizeof(float **) * input_layer_num);
  for (int32_t i = 0; i < input_layer_num; i++) {
    VALUE vinput_layer = rb_ary_entry(vinput_layers, i);
    VALUE vname =
        rb_hash_aref(vinput_layer, rb_to_symbol(rb_str_new2("name")));
    VALUE vdims =
        rb_hash_aref(vinput_layer, rb_to_symbol(rb_str_new2("dims")));
    int32_t dims_length =
        NUM2INT(rb_funcall(vdims, rb_intern("length"), 0, NULL));

    // prepare input buffer
    int32_t buffer_length = 1;
    for (int32_t j = 0; j < dims_length; j++)
      buffer_length *= NUM2INT(rb_ary_entry(vdims, j));

    float *input_buff = (float *)ruby_xmalloc(sizeof(float) * buffer_length);
    getModel(self)->input_buffs[i] = input_buff;
    ERROR_CHECK(
        menoh_model_builder_attach_external_buffer(
            getModel(self)->model_builder, StringValueCStr(vname), input_buff),
        rb_eStandardError);
  }

  // build model
  ERROR_CHECK(menoh_build_model(
                  getModel(self)->model_builder, getModel(self)->model_data,
                  StringValueCStr(vbackend), "", &(getModel(self)->model)),
              rb_eStandardError);

  return Qnil;
}

static VALUE wrap_model_run(VALUE self, VALUE dataset) {
  VALUE vbackend = getModel(self)->vbackend;
  VALUE vinput_layers = getModel(self)->vinput_layers;
  VALUE voutput_layers = getModel(self)->voutput_layers;

  int32_t input_layer_num =
      NUM2INT(rb_funcall(vinput_layers, rb_intern("length"), 0, NULL));
  int32_t output_layer_num =
      NUM2INT(rb_funcall(voutput_layers, rb_intern("length"), 0, NULL));

  // Copy input image data to model's input array
  for (int32_t i = 0; i < input_layer_num; i++) {
    VALUE vinput_layer = rb_ary_entry(vinput_layers, i);
    VALUE vname = rb_hash_aref(vinput_layer, rb_to_symbol(rb_str_new2("name")));
    VALUE vdims = rb_hash_aref(vinput_layer, rb_to_symbol(rb_str_new2("dims")));
    int32_t dims_length =
        NUM2INT(rb_funcall(vdims, rb_intern("length"), 0, NULL));
    int32_t buffer_length = 1;
    for (int32_t j = 0; j < dims_length; j++)
      buffer_length *= NUM2INT(rb_ary_entry(vdims, j));

    VALUE data = rb_ary_entry(dataset, i);
    for (int32_t j = 0; j < buffer_length; j++) {
      getModel(self)->input_buffs[i][j] =
          (float)(NUM2DBL(rb_ary_entry(data, j)));
    }
  }

  // attach output buffer to model
  getModel(self)->output_buffs =
      (float **)ruby_xmalloc(sizeof(float *) * output_layer_num);
  for (int32_t i = 0; i < output_layer_num; i++) {
    VALUE voutput_layer = rb_ary_entry(voutput_layers, i);
    float *output_buff;
    ERROR_CHECK(menoh_model_get_variable_buffer_handle(
                    getModel(self)->model, StringValueCStr(voutput_layer),
                    (void **)&output_buff),
                rb_eStandardError);
    getModel(self)->output_buffs[i] = output_buff;
  }

  // run model
  ERROR_CHECK(menoh_model_run(getModel(self)->model), rb_eStandardError);

  // Get output
  VALUE results = rb_ary_new();
  for (int32_t output_layer_i = 0; output_layer_i < output_layer_num;
       output_layer_i++) {
    VALUE voutput_layer = rb_ary_entry(voutput_layers, output_layer_i);
    VALUE result_each = rb_hash_new();

    // get dimention of output layers
    int32_t dim_size;
    int32_t output_buffer_length = 1;
    ERROR_CHECK(menoh_variable_profile_table_get_dims_size(
                    getModel(self)->variable_profile_table,
                    StringValueCStr(voutput_layer), &(dim_size)),
                rb_eStandardError);
    VALUE vresult_shape = rb_ary_new();
    // get each size of dimention
    for (int32_t dim = 0; dim < dim_size; dim++) {
      int32_t size;
      ERROR_CHECK(menoh_variable_profile_table_get_dims_at(
                      getModel(self)->variable_profile_table,
                      StringValueCStr(voutput_layer), dim, &(size)),
                  rb_eStandardError);
      rb_ary_push(vresult_shape, INT2NUM(size));
      output_buffer_length *= size;
    }

    // Convert result to Ruby Array
    VALUE vresult_buffer = rb_ary_new();
    for (int32_t j = 0; j < output_buffer_length; j++) {
      float *output_buff;
      output_buff = getModel(self)->output_buffs[output_layer_i];
      rb_ary_push(vresult_buffer, DBL2NUM(*(output_buff + j)));
    }

    rb_hash_aset(result_each, rb_to_symbol(rb_str_new2("name")), voutput_layer);
    rb_hash_aset(result_each, rb_to_symbol(rb_str_new2("shape")),
                 vresult_shape);
    rb_hash_aset(result_each, rb_to_symbol(rb_str_new2("data")),
                 vresult_buffer);
    rb_ary_push(results, result_each);
  }

  return results;
}

VALUE mMenoh;

void Init_menoh_native() {
  mMenoh = rb_define_module("Menoh");

  VALUE onnx = rb_define_class_under(mMenoh, "Menoh", rb_cObject);

  rb_define_alloc_func(onnx, wrap_menoh_alloc);
  rb_define_private_method(onnx, "native_init",
                           RUBY_METHOD_FUNC(wrap_menoh_init), 1);

  VALUE model = rb_define_class_under(mMenoh, "MenohModel", rb_cObject);

  rb_define_alloc_func(model, wrap_model_alloc);
  rb_define_private_method(model, "native_init",
                           RUBY_METHOD_FUNC(wrap_model_init), 2);

  rb_define_private_method(model, "native_run",
                           RUBY_METHOD_FUNC(wrap_model_run), 1);
}
