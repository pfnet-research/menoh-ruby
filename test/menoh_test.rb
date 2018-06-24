require 'test_helper'

MNIST_IN_NAME = '139900320569040'.freeze
MNIST_OUT_NAME = '139898462888656'.freeze

class MenohTest < Minitest::Test
  def test_that_it_has_a_version_number
    refute_nil ::Menoh::VERSION
  end

  def test_menoh_basic_function
    onnx = Menoh::Menoh.new('example/data/mnist.onnx')
    assert_instance_of(Menoh::Menoh, onnx)
    model_condition = {
      backend: 'mkldnn'
    }
    model = onnx.make_model(model_condition)
    assert_instance_of(Menoh::MenohModel, model)
    input_condition = {
      channel_num: 1,
      height: 28,
      width: 28,
      input_layer: MNIST_IN_NAME,
      output_layers: [MNIST_OUT_NAME]
    }
    batchsize = 3
    imageset = (0..(batchsize - 1)).map { |_i| (0..(1 * 28 * 28 - 1)).to_a }
    inference_results = model.run(imageset, input_condition)
    assert_instance_of(Array, inference_results)
    assert_equal(batchsize, inference_results.length)
  end

  def test_menoh_new_should_throw_when_the_path_value_is_invalid
    assert_raises { Menoh::Menoh.new('invalid path') }
  end

  def test_make_model_should_throw_when_the_condition_is_invaild
    onnx = Menoh::Menoh.new('example/data/mnist.onnx')
    conditions = [
      {},
      {
        backend: 'invalid'
      }
    ]
    conditions.each do |condition|
      assert_raises { onnx.make_model(condition) }
    end
  end

  def test_model_run_should_throw_when_model_condition_is_invalid
    onnx = Menoh::Menoh.new('example/data/mnist.onnx')
    model_condition = {
      backend: 'mkldnn'
    }
    model = onnx.make_model(model_condition)
    imageset = (0..9).map { |_i| (0..(1 * 28 * 28 - 1)).to_a }
    input_condition = {
      channel_num: 1,
      height: 28,
      width: 28,
      input_layer: MNIST_IN_NAME,
      output_layers: ['invalid']
    }
    assert_raises { model.run(imageset, input_condition) }
  end

  def test_model_run_should_throw_when_input_condition_is_invalid
    onnx = Menoh::Menoh.new('example/data/mnist.onnx')
    model_condition = {
      backend: 'mkldnn'
    }
    model = onnx.make_model(model_condition)

    imageset = (0..9).map { |_i| (0..(1 * 28 * 28 - 1)).to_a }
    valid_condition = {
      channel_num: 1,
      height: 28,
      width: 28,
      input_layer: MNIST_IN_NAME,
      output_layers: [MNIST_OUT_NAME]
    }
    input_conditions = [
      ## invalid imageset
      [
        nil,
        valid_condition
      ],
      [
        [],
        valid_condition
      ],
      [
        78_953_278,
        valid_condition
      ],
      [
        '[invalid]',
        valid_condition
      ],
      ## nil of condition
      [
        imageset,
        {}
      ],
      ## partial nil of condition
      [
        imageset,
        {
          height: 28,
          width: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          width: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 28,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 28,
          input_layer: MNIST_IN_NAME
        }
      ],
      ## invalid type
      [
        imageset,
        {
          channel_num: '1',
          height: 28,
          width: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: '28',
          width: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: '28',
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 28,
          input_layer: 9_999_999,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: [9_999_999]
        }
      ],
      ## zero
      [
        imageset,
        {
          channel_num: 0,
          height: 28,
          width: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 0,
          width: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 0,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 28,
          input_layer: '',
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 28,
          input_layer: '',
          output_layers: ['']
        }
      ],
      ## invalid image size
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 189,
          input_layer: MNIST_IN_NAME,
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      ## onnx doesn't have name
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 28,
          input_layer: 'invalid',
          output_layers: [MNIST_OUT_NAME]
        }
      ],
      [
        imageset,
        {
          channel_num: 1,
          height: 28,
          width: 28,
          input_layer: MNIST_IN_NAME,
          output_layers: ['invalid']
        }
      ]
    ]
    input_conditions.each do |args|
      assert_raises { model.run(args[0], args[1]) }
    end
  end
end
