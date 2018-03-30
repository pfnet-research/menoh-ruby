require 'rmagick'
include Magick

require 'runx'

# load dataset
imagelist = [
  "./data/0.png",
  "./data/1.png",
  "./data/2.png",
  "./data/3.png",
  "./data/4.png",
  "./data/5.png",
  "./data/6.png",
  "./data/7.png",
  "./data/8.png",
  "./data/9.png",
]

# load ONNX file
onnx_obj = Runx::Runx.new("./data/mnist.onnx")

MNIST_IN_NAME = "139900320569040"
MNIST_OUT_NAME = "139898462888656"

# conditions for inference
condition = {
  :batch_size => imagelist.length,
  :channel_num => 1,
  :height => 28,
  :width => 28,
  :input_layer => MNIST_IN_NAME,
  :output_layers => [MNIST_OUT_NAME]
}

# make model for inference under 'condition'
model = onnx_obj.make_model(condition)

# prepare dataset
imageset = imagelist.map do |image_filepath|
  image = Image.read(image_filepath).first.resize_to_fill(condition[:width], condition[:height])
  image.export_pixels(0, 0, image.columns, image.rows, 'i').map { |pix| pix }.to_a
end

# execute inference
inference_results = model.inference(imageset)

categories = (0..9).to_a
TOP_K = 1
inference_results.zip(imagelist).each do |inference_result, image_filepath|
  # sort by score
  sorted_result = inference_result[MNIST_OUT_NAME].zip(categories).sort_by{|x| -x[0]}


  # display result
  sorted_result[0, TOP_K].each do |score, category|
    puts "#{image_filepath} = #{category} : #{score}"
  end
end
