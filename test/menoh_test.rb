require 'test_helper'

MNIST_ONNX_FILE = 'example/data/mnist.onnx'.freeze
MNIST_IN_NAME = '139900320569040'.freeze
MNIST_OUT_NAME = '139898462888656'.freeze

class MenohTest < Minitest::Test
  def test_that_it_has_a_version_number
    refute_nil ::Menoh::VERSION
  end

  def test_menoh_basic_function
    onnx = Menoh::Menoh.new(MNIST_ONNX_FILE)
    assert_instance_of(Menoh::Menoh, onnx)
    batch_size = 3
    model_opt = {
      backend: 'mkldnn',
      input_layers: [
        {
          name: MNIST_IN_NAME,
          dims: [batch_size, 1, 28, 28]
        }
      ],
      output_layers: [MNIST_OUT_NAME]
    }
    model = onnx.make_model(model_opt)
    assert_instance_of(Menoh::MenohModel, model)
    10.times do
      imageset = [
        {
          name: MNIST_IN_NAME,
          data: (0..(batch_size - 1)).map { |_i| (0..(1 * 28 * 28 - 1)).to_a }.flatten
        }
      ]
      inferenced_results = model.run imageset
      assert_instance_of(Array, inferenced_results)
      assert_equal(MNIST_OUT_NAME, inferenced_results.first[:name])
      assert_equal(batch_size, inferenced_results.first[:data].length)
    end
  end

  def test_menoh_basic_function_with_block
    batch_size = 3
    model_opt = {
      backend: 'mkldnn',
      input_layers: [
        {
          name: MNIST_IN_NAME,
          dims: [batch_size, 1, 28, 28]
        }
      ],
      output_layers: [MNIST_OUT_NAME]
    }
    imageset = [
      {
        name: MNIST_IN_NAME,
        data: (0..(batch_size - 1)).map { |_i| (0..(1 * 28 * 28 - 1)).to_a }.flatten
      }
    ]
    Menoh::Menoh.new(MNIST_ONNX_FILE) do |onnx|
      assert_instance_of(Menoh::Menoh, onnx)
      onnx.make_model(model_opt) do |model|
        assert_instance_of(Menoh::MenohModel, model)
        model.run(imageset) do |inferenced_results|
          assert_instance_of(Array, inferenced_results)
          assert_equal(MNIST_OUT_NAME, inferenced_results.first[:name])
          assert_equal(batch_size, inferenced_results.first[:data].length)
        end
      end
    end
  end

  def test_menoh_new_should_throw_when_the_path_value_is_invalid
    assert_raises { Menoh::Menoh.new('invalid path') }
  end

  # def test_make_model_should_throw_when_the_option_is_invaild
  #   onnx = Menoh::Menoh.new(MNIST_ONNX_FILE)

  #   # empty
  #   assert_raises { onnx.make_model({}) }

  #   # missing
  #   missing_base_opt = {
  #     backend: 'mkldnn',
  #     batch_size: 1,
  #     channel_num: 1,
  #     height: 28,
  #     width: 28,
  #     input_layers: [MNIST_IN_NAME],
  #     output_layers: [MNIST_OUT_NAME]
  #   }
  #   missing_base_opt.keys do |key|
  #     opt = missing_base_opt.delete(key)
  #     assert_raises { onnx.make_model(opt) }
  #   end

  #   ## invalid type
  #   invalid_opts = [
  #     {
  #       backend: 'invalid',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: '1',
  #       channel_num: 1,
  #       height: 28,
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: '1',
  #       height: 28,
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: '28',
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: '28',
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     # {
  #     #   backend: 'mkldnn',
  #     #   batch_size: 1,
  #     #   channel_num: 1,
  #     #   height: 28,
  #     #   width: 28,
  #     #   input_layers: [9_999_999],
  #     #   output_layers: [MNIST_OUT_NAME]
  #     # },
  #     # {
  #     #   backend: 'mkldnn',
  #     #   batch_size: 1,
  #     #   channel_num: 1,
  #     #   height: 28,
  #     #   width: 28,
  #     #   input_layers: [MNIST_IN_NAME],
  #     #   output_layers: [9_999_999]
  #     # }
  #   ]
  #   invalid_opts.each do |opt|
  #     assert_raises { onnx.make_model(opt) }
  #   end

  #   ## zero
  #   zero_opts = [
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 0,
  #       channel_num: 1,
  #       height: 28,
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 0,
  #       height: 28,
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 0,
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: 0,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     }
  #   ]
  #   zero_opts.each do |opt|
  #     assert_raises { onnx.make_model(opt) }
  #   end

  #   empty_string_opts = [
  #     {
  #       backend: '',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: 28,
  #       input_layers: [''],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: 28,
  #       input_layers: [''],
  #       output_layers: ['']
  #     }
  #   ]
  #   empty_string_opts.each do |opt|
  #     assert_raises { onnx.make_model(opt) }
  #   end

  #   etc_opts = [
  #     ## invalid image size
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: 189,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     ## onnx doesn't have name
  #     {
  #       backend: 'mkldnn',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: 28,
  #       input_layers: ['invalid'],
  #       output_layers: [MNIST_OUT_NAME]
  #     },
  #     {
  #       backend: 'invalid',
  #       batch_size: 1,
  #       channel_num: 1,
  #       height: 28,
  #       width: 28,
  #       input_layers: [MNIST_IN_NAME],
  #       output_layers: ['invalid']
  #     }
  #   ]
  #   etc_opts.each do |opt|
  #     assert_raises { onnx.make_model(opt) }
  #   end
  # end

  # def test_model_run_should_throw_when_input_option_is_invalid
  #   onnx = Menoh::Menoh.new(MNIST_ONNX_FILE)
  #   batch_size = 10
  #   model_opt = {
  #     backend: 'mkldnn',
  #     batch_size: batch_size,
  #     channel_num: 1,
  #     height: 28,
  #     width: 28,
  #     input_layers: [MNIST_IN_NAME],
  #     output_layers: [MNIST_OUT_NAME]
  #   }
  #   model = onnx.make_model(model_opt)

  #   test_imagesets = [
  #     ## invalid imageset
  #     nil,
  #     [],
  #     78_953_278,
  #     '[invalid]'
  #   ]
  #   test_imagesets.each do |test_imageset|
  #     imageset = {
  #       MNIST_IN_NAME => test_imageset
  #     }
  #     assert_raises { model.run(imageset) }
  #   end
  # end

  # TODO: test_model_run_shold_throw_when_image_size_is_not_valid
end
