#include "menoh_ruby.h"
#include <ruby/thread.h>

static VALUE eError;
static VALUE eStdError;
static VALUE eUnknownError;
static VALUE eInvalidFilename;
static VALUE eUnsupportedONNXOpsetVersion;
static VALUE eONNXParseError;
static VALUE eInvalidDType;
static VALUE eInvalidAttributeType;
static VALUE eUnsupportedOperatorAttribute;
static VALUE eDimensionMismatch;
static VALUE eVariableNotFound;
static VALUE eIndexOutOfRange;
static VALUE eJSONParseError;
static VALUE eInvalidBackendName;
static VALUE eUnsupportedOperator;
static VALUE eFailedToConfigureOperator;
static VALUE eBackendError;
static VALUE eSameNamedVariableAlreadyExist;
static VALUE eUnsupportedInputDims;
static VALUE eSameNamedParameterAlreadyExist;
static VALUE eSameNamedAttributeAlreadyExist;
static VALUE eInvalidBackendConfigError;
static VALUE eInputNotFoundError;
static VALUE eOutputNotFoundError;

static void error_check(menoh_error_code ec) {
  VALUE e = eError; // eUnknownError might be better?

  switch (ec) {
  case menoh_error_code_success:
    return;
  case menoh_error_code_std_error:
    e = eStdError;
    break;
  case menoh_error_code_unknown_error:
    e = eUnknownError;
    break;
  case menoh_error_code_invalid_filename:
    e = eInvalidFilename;
    break;
  case menoh_error_code_unsupported_onnx_opset_version:
    e = eUnsupportedONNXOpsetVersion;
    break;
  case menoh_error_code_onnx_parse_error:
    e = eONNXParseError;
    break;
  case menoh_error_code_invalid_dtype:
    e = eInvalidDType;
    break;
  case menoh_error_code_invalid_attribute_type:
    e = eInvalidAttributeType;
    break;
  case menoh_error_code_unsupported_operator_attribute:
    e = eUnsupportedOperatorAttribute;
    break;
  case menoh_error_code_dimension_mismatch:
    e = eDimensionMismatch;
    break;
  case menoh_error_code_variable_not_found:
    e = eVariableNotFound;
    break;
  case menoh_error_code_index_out_of_range:
    e = eIndexOutOfRange;
    break;
  case menoh_error_code_json_parse_error:
    e = eJSONParseError;
    break;
  case menoh_error_code_invalid_backend_name:
    e = eInvalidBackendName;
    break;
  case menoh_error_code_unsupported_operator:
    e = eUnsupportedOperator;
    break;
  case menoh_error_code_failed_to_configure_operator:
    e = eFailedToConfigureOperator;
    break;
  case menoh_error_code_backend_error:
    e = eBackendError;
    break;
  case menoh_error_code_same_named_variable_already_exist:
    e = eSameNamedVariableAlreadyExist;
    break;
  case menoh_error_code_unsupported_input_dims:
    e = eUnsupportedInputDims;
    break;
  case menoh_error_code_same_named_parameter_already_exist:
    e = eSameNamedParameterAlreadyExist;
    break;
  case menoh_error_code_same_named_attribute_already_exist:
    e = eSameNamedAttributeAlreadyExist;
    break;
  case menoh_error_code_invalid_backend_config_error:
    e = eInvalidBackendConfigError;
    break;
  case menoh_error_code_input_not_found_error:
    e = eInputNotFoundError;
    break;
  case menoh_error_code_output_not_found_error:
    e = eOutputNotFoundError;
    break;
  }

  rb_raise(e, "%s", menoh_get_last_error_message());
}

#define ERROR_CHECK(statement) error_check(statement)


typedef struct menoh_ruby {
  menoh_model_data_handle model_data;
} menoh_ruby;

static void wrap_menoh_free(menoh_ruby *);

static const rb_data_type_t menoh_ruby_data_type = {
  "Menoh::Menoh",
  {NULL, (void(*)(void*))wrap_menoh_free, NULL,},
  0, NULL, RUBY_TYPED_FREE_IMMEDIATELY
};

static ID id_backend, id_input_layers, id_output_layers;
static ID id_data, id_dims, id_length, id_name, id_shape;

static menoh_ruby *getONNX(VALUE self) {
  menoh_ruby *p;
  TypedData_Get_Struct(self, menoh_ruby, &menoh_ruby_data_type, p);
  return p;
}

static void wrap_menoh_free(menoh_ruby *p) {
  menoh_delete_model_data(p->model_data);
  ruby_xfree(p);
}

static VALUE wrap_menoh_alloc(VALUE klass) {
  void *p = ruby_xmalloc(sizeof(menoh_ruby));
  memset(p, 0, sizeof(menoh_ruby));
  return TypedData_Wrap_Struct(klass, &menoh_ruby_data_type, p);
}

static VALUE wrap_menoh_init(VALUE self, VALUE vfilename) {
  FilePathValue(vfilename);
  char *filename = StringValueCStr(vfilename);

  // Load ONNX model
  menoh_model_data_handle model_data;
  ERROR_CHECK(menoh_make_model_data_from_onnx(filename, &model_data));
  getONNX(self)->model_data = model_data;

  return Qnil;
}

typedef struct menohModel {
  float **input_buffs;
  float **output_buffs;
  menoh_variable_profile_table_handle variable_profile_table;
  menoh_model_handle model;
  VALUE vinput_layers;
  VALUE voutput_layers;
  int32_t input_layer_num;
} menohModel;

static void wrap_model_free(menohModel *);
static void wrap_model_mark(menohModel *);

static const rb_data_type_t menohModel_data_type = {
  "Menoh::MenohModel",
  {(void(*)(void*))wrap_model_mark, (void(*)(void*))wrap_model_free, NULL,},
  0, NULL, RUBY_TYPED_FREE_IMMEDIATELY
};

static menohModel *getModel(VALUE self) {
  menohModel *p;
  TypedData_Get_Struct(self, menohModel, &menohModel_data_type, p);
  return p;
}

static void wrap_model_free(menohModel *p) {
  menoh_delete_variable_profile_table(p->variable_profile_table);
  menoh_delete_model(p->model);
  ruby_xfree(p->input_buffs);
  ruby_xfree(p->output_buffs);
  ruby_xfree(p);
}

static void wrap_model_mark(menohModel *p) {
  rb_gc_mark(p->vinput_layers);
  rb_gc_mark(p->voutput_layers);
}

static VALUE wrap_model_alloc(VALUE klass) {
  void *p = ruby_xmalloc(sizeof(menohModel));
  memset(p, 0, sizeof(menohModel));
  return TypedData_Wrap_Struct(klass, &menohModel_data_type, p);
}

struct build_vpt_arg {
  VALUE self;
  VALUE vinput_layers;
  VALUE voutput_layers;
  menoh_model_data_handle model_data;
  menoh_variable_profile_table_builder_handle vpt_builder;
};

static VALUE build_vpt(VALUE arg) {
  struct build_vpt_arg *arg2 = (struct build_vpt_arg*)arg;
  VALUE self = arg2->self;
  VALUE vinput_layers = arg2->vinput_layers;
  VALUE voutput_layers = arg2->voutput_layers;
  menoh_model_data_handle model_data = arg2->model_data;
  menoh_variable_profile_table_builder_handle vpt_builder = arg2->vpt_builder;

  // set output_layer
  int32_t output_layer_num =
      NUM2INT(rb_funcall(voutput_layers, id_length, 0));
  for (int32_t i = 0; i < output_layer_num; i++) {
    VALUE voutput_layer = rb_ary_entry(voutput_layers, i);
    ERROR_CHECK(menoh_variable_profile_table_builder_add_output_name(
                    vpt_builder, StringValueCStr(voutput_layer)));
  }

  // set input layer
  int32_t input_layer_num =
      NUM2INT(rb_funcall(vinput_layers, id_length, 0));
  getModel(self)->input_layer_num = input_layer_num;
  for (int32_t i = 0; i < input_layer_num; i++) {
    VALUE vinput_layer = rb_ary_entry(vinput_layers, i);
    VALUE vname = rb_hash_aref(vinput_layer, ID2SYM(id_name));
    VALUE vdims = rb_hash_aref(vinput_layer, ID2SYM(id_dims));
    int32_t dims_length =
        NUM2INT(rb_funcall(vdims, id_length, 0));

    int32_t *dims = (int32_t *)alloca(sizeof(int32_t) * dims_length);
    for (int32_t j = 0; j < dims_length; j++){
      dims[j] = NUM2INT(rb_ary_entry(vdims, j));
    }
    ERROR_CHECK(
        menoh_variable_profile_table_builder_add_input_profile(
            vpt_builder, StringValueCStr(vname),
            menoh_dtype_float,
            dims_length,
            dims));
  }

  // build variable profile table
  menoh_variable_profile_table_handle variable_profile_table;
  ERROR_CHECK(menoh_build_variable_profile_table(
                  vpt_builder, model_data,
                  &variable_profile_table));

  return (VALUE)variable_profile_table;
}

static VALUE vpt_builder_free(VALUE arg) {
  menoh_delete_variable_profile_table_builder((menoh_variable_profile_table_builder_handle)arg);
  return Qnil;
}

struct build_model_arg {
  VALUE self;
  VALUE vinput_layers;
  VALUE voutput_layers;
  VALUE vbackend;
  menoh_model_data_handle model_data;
  menoh_model_builder_handle model_builder;
};

static VALUE model_builder_free(VALUE arg) {
  menoh_delete_model_builder((menoh_model_builder_handle)arg);
  return Qnil;
}

static VALUE build_model(VALUE arg) {
  struct build_model_arg *arg2 = (struct build_model_arg*)arg;
  VALUE self = arg2->self;
  VALUE vinput_layers = arg2->vinput_layers;
  VALUE voutput_layers = arg2->voutput_layers;
  VALUE vbackend = arg2->vbackend;
  menoh_model_data_handle model_data = arg2->model_data;
  menoh_model_builder_handle model_builder = arg2->model_builder;

  // build model
  ERROR_CHECK(menoh_build_model(
                  model_builder, model_data,
                  StringValueCStr(vbackend), "", &getModel(self)->model));
  menoh_model_handle model = getModel(self)->model;

  // attach input buffer to model builder
  int32_t input_layer_num =
      NUM2INT(rb_funcall(vinput_layers, id_length, 0));
  getModel(self)->input_buffs =
      (float **)ruby_xmalloc(sizeof(float *) * input_layer_num);
  for (int32_t i = 0; i < input_layer_num; i++) {
    VALUE vinput_layer = rb_ary_entry(vinput_layers, i);
    VALUE vname =
        rb_hash_aref(vinput_layer, ID2SYM(id_name));
    VALUE vdims =
        rb_hash_aref(vinput_layer, ID2SYM(id_dims));
    int32_t dims_length =
        NUM2INT(rb_funcall(vdims, id_length, 0));

    // prepare input buffer
    int32_t buffer_length = 1;
    for (int32_t j = 0; j < dims_length; j++)
      buffer_length *= NUM2INT(rb_ary_entry(vdims, j));

    float *input_buff;
    ERROR_CHECK(menoh_model_get_variable_buffer_handle(
                    model, StringValueCStr(vname),
                    (void **)&input_buff));
    getModel(self)->input_buffs[i] = input_buff;
  }

  // attach output buffer to model
  int32_t output_layer_num =
      NUM2INT(rb_funcall(voutput_layers, id_length, 0));
  getModel(self)->output_buffs =
      (float **)ruby_xmalloc(sizeof(float *) * output_layer_num);
  for (int32_t i = 0; i < output_layer_num; i++) {
    VALUE voutput_layer = rb_ary_entry(voutput_layers, i);
    float *output_buff;
    ERROR_CHECK(menoh_model_get_variable_buffer_handle(
                    model, StringValueCStr(voutput_layer),
                    (void **)&output_buff));
    getModel(self)->output_buffs[i] = output_buff;
  }

  return Qnil;
}

static VALUE wrap_model_init(VALUE self, VALUE vonnx, VALUE option) {
  // option
  menoh_model_data_handle model_data = getONNX(vonnx)->model_data;
  VALUE vbackend = rb_hash_aref(option, ID2SYM(id_backend));

  // option
  VALUE vinput_layers =
      rb_hash_aref(option, ID2SYM(id_input_layers));
  VALUE voutput_layers =
      rb_hash_aref(option, ID2SYM(id_output_layers));
  getModel(self)->vinput_layers = vinput_layers;
  getModel(self)->voutput_layers = voutput_layers;

  // get vpt builder
  menoh_variable_profile_table_builder_handle vpt_builder;
  ERROR_CHECK(
      menoh_make_variable_profile_table_builder(&vpt_builder));

  // build variable profile table
  struct build_vpt_arg build_vpt_arg = {
    .self = self,
    .vinput_layers = vinput_layers,
    .voutput_layers = voutput_layers,
    .model_data = model_data,
    .vpt_builder = vpt_builder
  };
  getModel(self)->variable_profile_table = (menoh_variable_profile_table_handle)
    rb_ensure(build_vpt, (VALUE)&build_vpt_arg, vpt_builder_free, (VALUE)vpt_builder);

  // optimize
  ERROR_CHECK(
      menoh_model_data_optimize(model_data,
                                getModel(self)->variable_profile_table));

  // get model buildler
  menoh_model_builder_handle model_builder;
  ERROR_CHECK(menoh_make_model_builder(getModel(self)->variable_profile_table,
                                        &model_builder));

  // build model
  struct build_model_arg build_model_arg = {
    .self = self,
    .vinput_layers = vinput_layers,
    .voutput_layers = voutput_layers,
    .vbackend = vbackend,
    .model_data = model_data,
    .model_builder = model_builder
  };
  rb_ensure(build_model, (VALUE)&build_model_arg, model_builder_free, (VALUE)model_builder);

  return Qnil;
}


struct model_run_arg {
  menoh_model_handle model;
  menoh_error_code err;
};

static void *model_run(void *arg) {
  struct model_run_arg* arg2 = (struct model_run_arg*)arg;
  arg2->err = menoh_model_run(arg2->model);
  return NULL;
}

static VALUE wrap_model_run(VALUE self, VALUE dataset) {
  VALUE vinput_layers = getModel(self)->vinput_layers;
  VALUE voutput_layers = getModel(self)->voutput_layers;

  int32_t input_layer_num =
      NUM2INT(rb_funcall(vinput_layers, id_length, 0));
  int32_t output_layer_num =
      NUM2INT(rb_funcall(voutput_layers, id_length, 0));

  // Copy input image data to model's input array
  for (int32_t i = 0; i < input_layer_num; i++) {
    VALUE vinput_layer = rb_ary_entry(vinput_layers, i);
    VALUE vdims = rb_hash_aref(vinput_layer, ID2SYM(id_dims));
    int32_t dims_length =
        NUM2INT(rb_funcall(vdims, id_length, 0));
    int32_t buffer_length = 1;
    for (int32_t j = 0; j < dims_length; j++)
      buffer_length *= NUM2INT(rb_ary_entry(vdims, j));

    VALUE data = rb_ary_entry(dataset, i);
    for (int32_t j = 0; j < buffer_length; j++) {
      getModel(self)->input_buffs[i][j] =
          (float)(NUM2DBL(rb_ary_entry(data, j)));
    }
  }

  // run model
  struct model_run_arg model_run_arg = {
    .model = getModel(self)->model,
    .err = menoh_error_code_success,
  };
  rb_thread_call_without_gvl(model_run, &model_run_arg, RUBY_UBF_IO, NULL);
  ERROR_CHECK(model_run_arg.err);

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
                    StringValueCStr(voutput_layer), &(dim_size)));
    VALUE vresult_shape = rb_ary_new();
    // get each size of dimention
    for (int32_t dim = 0; dim < dim_size; dim++) {
      int32_t size;
      ERROR_CHECK(menoh_variable_profile_table_get_dims_at(
                      getModel(self)->variable_profile_table,
                      StringValueCStr(voutput_layer), dim, &(size)));
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

    rb_hash_aset(result_each, ID2SYM(id_name), voutput_layer);
    rb_hash_aset(result_each, ID2SYM(id_shape),
                 vresult_shape);
    rb_hash_aset(result_each, ID2SYM(id_data),
                 vresult_buffer);
    rb_ary_push(results, result_each);
  }

  return results;
}

void Init_menoh_native() {
  id_backend = rb_intern("backend");
  id_input_layers = rb_intern("input_layers");
  id_output_layers = rb_intern("output_layers");
  id_data = rb_intern("data");
  id_dims = rb_intern("dims");
  id_length = rb_intern("length");
  id_name = rb_intern("name");
  id_shape = rb_intern("shape");

  VALUE mMenoh = rb_define_module("Menoh");

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

  eError                          = rb_define_class_under(mMenoh, "Error", rb_eStandardError);
  eStdError                       = rb_define_class_under(mMenoh, "StdError", eError);
  eUnknownError                   = rb_define_class_under(mMenoh, "UnknownError", eError);
  eInvalidFilename                = rb_define_class_under(mMenoh, "InvalidFilename", eError);
  eUnsupportedONNXOpsetVersion    = rb_define_class_under(mMenoh, "UnsupportedONNXOpsetVersion", eError);
  eONNXParseError                 = rb_define_class_under(mMenoh, "ONNXParseError", eError);
  eInvalidDType                   = rb_define_class_under(mMenoh, "InvalidDType", eError);
  eInvalidAttributeType           = rb_define_class_under(mMenoh, "InvalidAttributeType", eError);
  eUnsupportedOperatorAttribute   = rb_define_class_under(mMenoh, "UnsupportedOperatorAttribute", eError);
  eDimensionMismatch              = rb_define_class_under(mMenoh, "DimensionMismatch", eError);
  eVariableNotFound               = rb_define_class_under(mMenoh, "VariableNotFound", eError);
  eIndexOutOfRange                = rb_define_class_under(mMenoh, "IndexOutOfRange", eError);
  eJSONParseError                 = rb_define_class_under(mMenoh, "JSONParseError", eError);
  eInvalidBackendName             = rb_define_class_under(mMenoh, "InvalidBackendName", eError);
  eUnsupportedOperator            = rb_define_class_under(mMenoh, "UnsupportedOperator", eError);
  eFailedToConfigureOperator      = rb_define_class_under(mMenoh, "FailedToConfigureOperator", eError);
  eBackendError                   = rb_define_class_under(mMenoh, "BackendError", eError);
  eSameNamedVariableAlreadyExist  = rb_define_class_under(mMenoh, "SameNamedVariableAlreadyExist", eError);
  eUnsupportedInputDims           = rb_define_class_under(mMenoh, "UnsupportedInputDims", eError);
  eSameNamedParameterAlreadyExist = rb_define_class_under(mMenoh, "SameNamedParameterAlreadyExist", eError);
  eSameNamedAttributeAlreadyExist = rb_define_class_under(mMenoh, "SameNamedAttributeAlreadyExist", eError);
  eInvalidBackendConfigError      = rb_define_class_under(mMenoh, "InvalidBackendConfigError", eError);
  eInputNotFoundError             = rb_define_class_under(mMenoh, "InputNotFoundError", eError);
  eOutputNotFoundError            = rb_define_class_under(mMenoh, "OutputNotFoundError", eError);
}
