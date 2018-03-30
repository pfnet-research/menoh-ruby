# Tutorial

In this tutorial, we are going to make a CNN model inference software.
This tutorial is based on [Runx](https://github.com/pfnet-research/runxpfnet-research)'s [original tutorial](TODO).

This script loads `data/VGG16.onnx` and takes input image, then outputs classification result.

## Preprocessing input

First of all, preprocessing input is required. `data/VGG16.onnx` takes 3 channels 224 x 224 sized image but input image is not always sized 224x224. So we use Imagemagick's `resize_to_fill` method for resizing.

Runx takes images as NCHW format (N x Channels x Height x Width), but `Mat` of OpenCV holds image as HWC format (Height x Width x Channels). In addition, `data/VGG16.onnx` takes RGB image but `Mat` holds image as BGR format. So next we call `export_pixels` method for each channels `["R", "G", "B"]`, then `flatten` arrays.

```ruby
imagelist = [
  "./data/Light_sussex_hen.jpg",
  "./data/honda_nsx.jpg",
]
imageset = imagelist.map do |image_filepath|
  image = Image.read(image_filepath).first.resize_to_fill(condition[:width], condition[:height])
  "RGB".split('').map do |color|
    image.export_pixels(0, 0, image.columns, image.rows, color).map { |pix| pix }.to_a
  end.flatten
end
```

In current case, the range of pixel value `data/VGG16.onnx` taking is \f$[0, 256)\f$ and it matches the range of `Mat`. So we have not to scale the values now.

However, sometimes model takes values scaled in range \f$[0.0, 1.0]\f$ or something. In that case, we can scale values here:

```ruby
imageset = imagelist.map do |image_filepath|
  image = Image.read(image_filepath).first.resize_to_fill(condition[:width], condition[:height])
  "RGB".split('').map do |color|
    image.export_pixels(0, 0, image.columns, image.rows, color).map { |pix| pix/257 }.to_a
  end.flatten
end
```

## Setup model

For gettinig ONNX model's named variables, please refer to [Runx Tutorial](TODO tutorial.md).

VGG16 has one input and one output. So now we can check that the input name is *140326425860192* (input of 0:Conv) and the output name is *140326200803680* (output of 39:Softmax).

Some of we are interested the feature vector of input image. So in addition, we are going to take the output of 32:FC(fc6, which is the first FC layer after CNNs) named *140326200777584*.

We define name aliases for convenience:

```ruby
CONV1_1_IN_NAME = "140326425860192"
FC6_OUT_NAME = "140326200777584"
SOFTMAX_OUT_NAME = "140326200803680"
```

To build model, we load model data from ONNX file:

```ruby
onnx_obj = Runx::Runx.new("./data/VGG16.onnx")
```

Now let's build the model.

```ruby
condition = {
  :batch_size => imagelist.length,
  :channel_num => 3,
  :height => 224,
  :width => 224,
  :input_layer => CONV1_1_IN_NAME,
  :output_layers => [FC6_OUT_NAME, SOFTMAX_OUT_NAME]
}
model = onnx_obj.make_model(condition)
```

## Run inference and get result

Now we can run inference.

```ruby
# execute inference
inference_results = model.inference(imageset)
```

The `inference_results` is the array that contains the hash of results of `output_layers`. So you can get each value as follows.

```ruby
inference_results.each do |inference_result|
  fc6_out = inference_result[FC6_OUT_NAME]
  softmax_out = inference_result[SOFTMAX_OUT_NAME]
end
```

That's it.

The full code is available at [VGG16 example](TODO/example/example_vgg16.rb).
