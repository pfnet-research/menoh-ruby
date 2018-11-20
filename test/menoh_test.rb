require 'test_helper'
require 'numo/narray'

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

  def test_menoh_basic_function_numo
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
          data: Numo::SFloat.zeros(batch_size, 1, 28, 28)
        }
      ]
      inferenced_results = model.run_numo imageset
      assert_instance_of(Hash, inferenced_results)
      assert_instance_of(Numo::SFloat, inferenced_results[MNIST_OUT_NAME])
      assert_equal([batch_size, 10], inferenced_results[MNIST_OUT_NAME].shape)
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

  def test_make_model_should_throw_when_the_option_is_invaild
    onnx = Menoh::Menoh.new(MNIST_ONNX_FILE)

    # empty
    assert_raises { onnx.make_model({}) }

    # missing
    missing_base_opt = {
      backend: 'mkldnn',
      input_layers: [
        {
          name: MNIST_IN_NAME,
          dims: [10, 1, 28, 28]
        }
      ],
      output_layers: [MNIST_OUT_NAME]
    }
    missing_base_opt.keys do |key|
      opt = missing_base_opt.delete(key)
      assert_raises { onnx.make_model(opt) }
    end

    ## invalid type
    invalid_opts = [
      {
        backend: 'invalid',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [10, 1, 28, 28]
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: 'invalid',
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            namee: MNIST_IN_NAME,
            dims: [10, 1, 28, 28]
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dimss: [10, 1, 28, 28]
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: 'invalid',
            dims: [10, 1, 28, 28]
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [10, 'invalid']
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [10, 1, 28, 28]
          }
        ],
        output_layers: ['invalid']
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [10, 1, 28, 28],
            dtype: 'invalid'
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      }
    ]
    invalid_opts.each do |opt|
      assert_raises { onnx.make_model(opt) }
    end

    ## zero or empty
    zero_opts = [
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: 0,
            dims: [10, 1, 28, 28]
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: []
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [0, 0, 0, 0]
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [0, 1, 0, 0]
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [10, 1, 28, 28]
          }
        ],
        output_layers: []
      }
    ]
    zero_opts.each do |opt|
      assert_raises { onnx.make_model(opt) }
    end

    etc_opts = [
      ## invalid image size
      # # omit this test because Menoh accepts any size
      # {
      #   backend: 'mkldnn',
      #   input_layers: [
      #     {
      #       name: MNIST_IN_NAME,
      #       dims: [10, 1, 28, 128]
      #     }
      #   ],
      #   output_layers: [MNIST_OUT_NAME]
      # },
      ## onnx doesn't have name
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: 'invalid',
            dims: [10, 1, 28, 28]
          }
        ],
        output_layers: [MNIST_OUT_NAME]
      },
      {
        backend: 'mkldnn',
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [10, 1, 28, 28]
          }
        ],
        output_layers: ['invalid']
      }
    ]
    # TODO: test for menoh_variable_profile_table_builder_add_input_profile_dims_2
    etc_opts.each do |opt|
      assert_raises { onnx.make_model(opt) }
    end
  end

  def test_model_run_should_throw_when_input_option_is_invalid
    onnx = Menoh::Menoh.new(MNIST_ONNX_FILE)
    batch_size = 10
    model_opt = {
      backend: 'mkldnn',
      input_layers: [
        {
          name: MNIST_IN_NAME,
          dims: [10, 1, 28, 28]
        }
      ],
      output_layers: [MNIST_OUT_NAME]
    }
    model = onnx.make_model(model_opt)

    test_imagesets = [
      ## invalid imageset
      nil,
      [],
      78_953_278,
      '[invalid]',
      (0..(batch_size - 1)).map { |_i| (0..(1 * 999 * 999 - 1)).to_a }.flatten
    ]
    test_imagesets.each do |test_imageset|
      imageset = {
        name: MNIST_IN_NAME,
        data: test_imageset
      }
      assert_raises { model.run(imageset) }
    end
  end

  def test_reshape
    assert_equal([[0], [1], [2], [3]], Menoh::Util.reshape([0, 1, 2, 3], [4, 1]))
    assert_equal([[0, 1], [2, 3]], Menoh::Util.reshape([0, 1, 2, 3], [2, 2]))
    assert_equal([[0, 1, 2, 3]], Menoh::Util.reshape([0, 1, 2, 3], [1, 4]))
    assert_equal([[[0, 1], [2, 3]]], Menoh::Util.reshape([0, 1, 2, 3], [1, 2, 2]))
    assert_equal([
                   [
                     [0, 1, 2, 3],
                     [4, 5, 6, 7]
                   ]
                 ],
                 Menoh::Util.reshape([0, 1, 2, 3, 4, 5, 6, 7],
                         [1, 2, 4]))
    assert_equal([
                   [
                     [0, 1],
                     [2, 3]
                   ],
                   [
                     [4, 5],
                     [6, 7]
                   ]
                 ],
                 Menoh::Util.reshape([0, 1, 2, 3, 4, 5, 6, 7],
                         [2, 2, 2]))
  end
end
