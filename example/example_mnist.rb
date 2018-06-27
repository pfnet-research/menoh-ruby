require 'rmagick'
require 'menoh'

# load dataset
image_list = [
  './data/0.png',
  './data/1.png',
  './data/2.png',
  './data/3.png',
  './data/4.png',
  './data/5.png',
  './data/6.png',
  './data/7.png',
  './data/8.png',
  './data/9.png'
]
input_shape = {
  channel_num: 1,
  width: 28,
  height: 28
}

# load ONNX file
onnx_obj = Menoh::Menoh.new './data/mnist.onnx'

# onnx variable name
MNIST_IN_NAME = '139900320569040'.freeze
MNIST_OUT_NAME = '139898462888656'.freeze

# model options for model
model_opt = {
  backend: 'mkldnn',
  input_layers: [
    {
      name: MNIST_IN_NAME,
      dims: [
        image_list.length,
        input_shape[:channel_num],
        input_shape[:width],
        input_shape[:height]
      ]
    }
  ],
  output_layers: [MNIST_OUT_NAME]
}
# make model for inference under 'model_opt'
model = onnx_obj.make_model model_opt

# prepare dataset
image_set = [
  {
    name: MNIST_IN_NAME,
    data: image_list.map do |image_filepath|
      image = Magick::Image.read(image_filepath).first
      image = image.resize_to_fill(input_shape[:width], input_shape[:height])
      image.export_pixels(0, 0, image.columns, image.rows, 'i').map { |pix| pix / 256 }
    end.flatten
  }
]
# execute inference
inferenced_results = model.run image_set

categories = (0..9).to_a
TOP_K = 1
layer_result = inferenced_results.find { |x| x[:name] == MNIST_OUT_NAME }
layer_result[:data].zip(image_list).each do |image_result, image_filepath|
  # sort by score
  sorted_result = image_result.zip(categories).sort_by { |x| -x[0] }

  # display result
  sorted_result[0, TOP_K].each do |score, category|
    puts "#{image_filepath} = #{category} : #{score}"
  end
end
