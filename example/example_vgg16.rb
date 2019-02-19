require 'open-uri'
require 'rmagick'
require 'menoh'

# download dependencies
def download_file(url, output)
  return if File.exist? output

  puts "downloading... #{url}"
  File.open(output, 'wb') do |f_output|
    open(url, 'rb') do |f_input|
      f_output.write f_input.read
    end
  end
end
download_file('https://preferredjp.box.com/shared/static/o2xip23e3f0knwc5ve78oderuglkf2wt.onnx', './data/VGG16.onnx')
download_file('https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt', './data/synset_words.txt')
download_file('https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg', './data/Light_sussex_hen.jpg')
download_file('https://upload.wikimedia.org/wikipedia/commons/f/fd/FoS20162016_0625_151036AA_%2827826100631%29.jpg', './data/honda_nsx.jpg')

# load dataset
image_list = [
  './data/Light_sussex_hen.jpg',
  './data/honda_nsx.jpg'
]
input_shape = {
  channel_num: 3,
  width: 224,
  height: 224
}
rgb_offset = {
  R: 123.68,
  G: 116.779,
  B: 103.939
}

# load ONNX file
onnx_obj = Menoh::Menoh.new './data/VGG16.onnx'

# onnx variable name
CONV1_1_IN_NAME = 'Input_0'.freeze
FC6_OUT_NAME = 'Gemm_0'.freeze
SOFTMAX_OUT_NAME = 'Softmax_0'.freeze

# model options for model
model_opt = {
  backend: 'mkldnn',
  input_layers: [
    {
      name: CONV1_1_IN_NAME,
      dims: [
        image_list.length,
        input_shape[:channel_num],
        input_shape[:height],
        input_shape[:width]
      ]
    }
  ],
  output_layers: [FC6_OUT_NAME, SOFTMAX_OUT_NAME]
}
# make model for inference under 'model_opt'
model = onnx_obj.make_model model_opt

# prepare dataset
image_set = [
  {
    name: CONV1_1_IN_NAME,
    data: image_list.map do |image_filepath|
      image = Magick::Image.read(image_filepath).first
      image = image.resize_to_fill(input_shape[:width], input_shape[:height])
      'RGB'.split('').map do |color|
        image.export_pixels(0, 0, image.columns, image.rows, color).map do |pix|
          pix / 256 - rgb_offset[color.to_sym]
        end
      end.flatten
    end.flatten
  }
]

# execute inference
inference_results = model.run image_set

# load category definition
categories = File.read('./data/synset_words.txt').split("\n")
TOP_K = 5
layer_result = inference_results.find { |x| x[:name] == SOFTMAX_OUT_NAME }
layer_result[:data].zip(image_list).each do |image_result, image_filepath|
  puts "=== Result for #{image_filepath} ==="

  # sort by score
  sorted_result = image_result.zip(categories).sort_by { |x| -x[0] }

  # display result
  sorted_result[0, TOP_K].each do |score, category|
    puts "#{category} : #{score}"
  end
end
