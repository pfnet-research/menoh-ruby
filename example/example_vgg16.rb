require 'open-uri'
require 'rmagick'
include Magick

require 'menoh'

# download dependencies
def download_file(url, output)
  return if File.exist? output
  puts "downloading... #{url}"
  open(output, 'wb') do |f_output|
    open(url, 'rb') do |f_input|
      f_output.write f_input.read
    end
  end
end
download_file('https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1', './data/VGG16.onnx')
download_file('https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt', './data/synset_words.txt')
download_file('https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg', './data/Light_sussex_hen.jpg')
download_file('https://upload.wikimedia.org/wikipedia/commons/f/fd/FoS20162016_0625_151036AA_%2827826100631%29.jpg', './data/honda_nsx.jpg')

# load dataset
imagelist = [
  './data/Light_sussex_hen.jpg',
  './data/honda_nsx.jpg'
]

# load ONNX file
onnx_obj = Menoh::Menoh.new './data/VGG16.onnx'

CONV1_1_IN_NAME = '140326425860192'.freeze
FC6_OUT_NAME = '140326200777584'.freeze
SOFTMAX_OUT_NAME = '140326200803680'.freeze

# conditions for model
model_condition = {
  backend: 'mkldnn'
}
# make model for inference under 'condition'
model = onnx_obj.make_model model_condition

# conditions for input
input_condition = {
  channel_num: 3,
  height: 224,
  width: 224,
  input_layer: CONV1_1_IN_NAME,
  output_layers: [FC6_OUT_NAME, SOFTMAX_OUT_NAME]
}
# prepare dataset
imageset = imagelist.map do |image_filepath|
  image = Image.read(image_filepath).first.resize_to_fill(input_condition[:width], input_condition[:height])
  'RGB'.split('').map do |color|
    image.export_pixels(0, 0, image.columns, image.rows, color).map { |pix| pix / 256 }
  end.flatten
end

# execute inference
inference_results = model.run imageset, input_condition

# load category definition
categories = File.read('./data/synset_words.txt').split("\n")

TOP_K = 5
inference_results.zip(imagelist).each do |inference_result, image_filepath|
  puts "=== Result for #{image_filepath} ==="

  # sort by score
  sorted_result = inference_result[SOFTMAX_OUT_NAME].zip(categories).sort_by { |x| -x[0] }

  # display result
  sorted_result[0, TOP_K].each do |score, category|
    puts "#{category} : #{score}"
  end
end
