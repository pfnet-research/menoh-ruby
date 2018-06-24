require 'rmagick'
include Magick

require 'menoh'

# load dataset
imagelist = [
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

# load ONNX file
onnx_obj = Menoh::Menoh.new './data/mnist.onnx'

# onnx variable name
MNIST_IN_NAME = '139900320569040'.freeze
MNIST_OUT_NAME = '139898462888656'.freeze

# conditions for model
model_condition = {
  backend: 'mkldnn'
}
# make model for inference under 'model_condition'
model = onnx_obj.make_model model_condition

# conditions for input
input_condition = {
  channel_num: 1,
  height: 28,
  width: 28,
  input_layer: MNIST_IN_NAME,
  output_layers: [MNIST_OUT_NAME]
}
# prepare dataset
imageset = imagelist.map do |image_filepath|
  image = Image.read(image_filepath).first.resize_to_fill(input_condition[:width], input_condition[:height])
  image.export_pixels(0, 0, image.columns, image.rows, 'i').map { |pix| pix / 256 }
end
# execute inference
model.run imageset, input_condition do |inference_results|
  categories = (0..9).to_a
  TOP_K = 1
  inference_results.zip(imagelist).each do |inference_result, image_filepath|
    # sort by score
    sorted_result = inference_result[MNIST_OUT_NAME].zip(categories).sort_by { |x| -x[0] }

    # display result
    sorted_result[0, TOP_K].each do |score, category|
      puts "#{image_filepath} = #{category} : #{score}"
    end
  end
end
