# Tutorial

In this tutorial, we are going to make a CNN model inference software.
This tutorial is based on [Menoh](https://github.com/pfnet-research/menoh)'s [original tutorial](https://github.com/pfnet-research/menoh/blob/master/docs/tutorial.md).

This script loads `data/VGG16.onnx` and takes input image, then outputs classification result.

## Setup model

For gettinig ONNX model's named variables, please refer to [Menoh Tutorial](https://github.com/pfnet-research/menoh/blob/master/docs/tutorial.md).

VGG16 has one input and one output. So now we can check that the input name is *140326425860192* (input of 0:Conv) and the output name is *140326200803680* (output of 39:Softmax).

Some of we are interested the feature vector of input image. So in addition, we are going to take the output of 32:FC(fc6, which is the first FC layer after CNNs) named *140326200777584*.

We define name aliases for convenience:

```ruby
CONV1_1_IN_NAME = '140326425860192'.freeze
FC6_OUT_NAME = '140326200777584'.freeze
SOFTMAX_OUT_NAME = '140326200803680'.freeze
```

To build model, we load model data from ONNX file:

```ruby
onnx_obj = Menoh::Menoh.new './data/VGG16.onnx'
```

Now let's build the model.

```ruby
# data shape of input images
input_shape = {
  channel_num: 3,
  width: 224,
  height: 224
}
# model options for model
model_opt = {
  backend: 'mkldnn',
  input_layers: [
    {
      name: CONV1_1_IN_NAME,
      dims: [
        image_list.length,
        input_shape[:channel_num],
        input_shape[:width],
        input_shape[:height]
      ]
    }
  ],
  output_layers: [FC6_OUT_NAME, SOFTMAX_OUT_NAME]
}
# make model for inference under 'model_opt'
model = onnx_obj.make_model model_opt
```

## Preprocessing dataset

Before running the inference, the preprocessing of input dataset is required. `data/VGG16.onnx` takes 3 channels 224 x 224 sized image but input image is not always sized 224x224. So we use Imagemagick's `resize_to_fill` method for resizing.

`VGG16.onnx`'s input layer *140326425860192* takes images as NCHW format (N x Channels x Height x Width). But RMagick's image array has alternately flatten values for each channel. So next we call `export_pixels` method for each channels `['B', 'G', 'R']`, then `flatten` arrays.

```ruby
image_list = [
  './data/Light_sussex_hen.jpg',
  './data/honda_nsx.jpg',
]
image_set = [
  {
    name: CONV1_1_IN_NAME,
    data: image_list.map do |image_filepath|
      image = Magick::Image.read(image_filepath).first
      image = image.resize_to_fill(input_shape[:width], input_shape[:height])
      'BGR'.split('').map do |color|
        image.export_pixels(0, 0, image.columns, image.rows, color).map { |pix| pix / 256 }
      end.flatten
    end.flatten
  }
]
```

In current case, the range of pixel value `data/VGG16.onnx` taking is [0, 256]. On the other hand RMagick's image array takes [0, 65536]. So we have to scale the values by dividing 256.

And sometimes model takes values scaled in range [0, 1] or something. In that case, we can scale values here:

```ruby
image_set = [
  {
    name: CONV1_1_IN_NAME,
    data: image_list.map do |image_filepath|
      image = Magick::Image.read(image_filepath).first
      image = image.resize_to_fill(input_shape[:width], input_shape[:height])
      'BGR'.split('').map do |color|
        image.export_pixels(0, 0, image.columns, image.rows, color).map { |pix| pix / 65536 }
      end.flatten
    end.flatten
  }
]
```

## Run inference and get results

Now we can run the inference.

```ruby
# execute inference
inferenced_results = model.run image_set
```

The `inferenced_results` is the array that contains the hash of results of `output_layers`. So you can get each value as follows.

```ruby
fc6_out = inferenced_results.find { |x| x[:name] == FC6_OUT_NAME }
softmax_out = inferenced_results.find { |x| x[:name] == SOFTMAX_OUT_NAME }
```

That's it.

The full code is available at [VGG16 example](https://github.com/pfnet-research/menoh-ruby/blob/master/example/example_vgg16.rb).
