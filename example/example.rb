require 'rmagick'
include Magick

require 'instant' # TODO

# load ONNX file
onnx_obj = Instant.new("VGG16.onnx") # TODO

CONV1_1_IN_NAME = "140326425860192"
FC6_OUT_NAME = "140326200777976"
SOFTMAX_OUT_NAME = "140326200803680"

# load dataset
imagelist = [
  "../data/Light_sussex_hen.jpg",
  "photo/dog.jpg",
  "photo/formula1.jpg",
  "photo/laptoppc.jpg",
  "photo/polarbear.jpg",
  "photo/vatican.jpg",
]

# conditions for inference
condition = {
  :batch_size => imagelist.length,
  :channel_num => 3,
  :height => 224,
  :width => 224,
  :input_layer => CONV1_1_IN_NAME,
  :output_layers => [FC6_OUT_NAME, SOFTMAX_OUT_NAME]
}

# prepare dataset
imageset = imagelist.map do |image_filepath|
  image = Image.read(image_filepath).first
  image.resize_to_fill(condition[:width], condition[:height])
end

# make model for inference under 'condition'
model = onnx_obj.make_model(condition)

# execute inference
inference_results = model.inference(imageset)

# load category definition
categories = File.read('../data/synset_words.txt').split("\n")

TOP_K = 5
inference_results.zip(imagelist).each do |inference_result, image_filepath|
  puts "=== Result for #{image_filepath} ==="

  # sort by score
  sorted_result = inference_result[SOFTMAX_OUT_NAME].zip(categories).sort_by{|x| -x[0]}

  # display result
  sorted_result[0, TOP_K].each do |score, category|
  puts "#{category} : #{score}"
  end
end
