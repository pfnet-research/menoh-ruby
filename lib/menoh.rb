require 'menoh/version'
require 'menoh/menoh_native'

module Menoh
  class Menoh
    def initialize(file)
      if !file.instance_of?(String) || !File.exist?(file)
        raise "No such file : #{file}"
      end

      native_init file
      yield self if block_given?
    end

    def make_model(option)
      if option[:backend].nil? || (option[:backend] != 'mkldnn')
        raise "Invalid ':backend' : #{option[:backend]}"
      end
      model = MenohModel.new self, option
      yield model if block_given?
      model
    end
  end
end

def transpose(buffer, shape)
  sliced_buffer = buffer.each_slice(buffer.length / shape[0]).to_a
  if shape.length > 2
    next_shape = shape.slice(1, a.length)
    sliced_buffer = sliced_buffer.map { |buf| transpose buf, next_shape }
  end
  sliced_buffer
end

module Menoh
  class MenohModel
    def initialize(menoh, option)
      if option[:input_layers].nil? || option[:input_layers].empty?
        raise 'Required : input_layers'
      end
      raise 'Required : input_layers' unless option[:input_layers].instance_of?(Array)
      option[:input_layers].each_with_index do |input_layer, i|
        raise 'Invalid option : input_layers' unless input_layer.instance_of?(Hash)
        raise "Need valid name for input_layer[#{i}]" unless input_layer[:name].instance_of?(String)
        raise "Need valid dims for input_layer[#{i}]" unless input_layer[:dims].instance_of?(Array)
      end
      if option[:output_layers].nil? || option[:output_layers].empty?
        raise "Invalid ':output_layers'"
      end
      native_init menoh, option
      @option = option
      yield self if block_given?
    end

    def run(dataset)
      raise 'Invalid dataset' if !dataset.instance_of?(Array) || dataset.empty?
      if dataset.length != @option[:input_layers].length
        raise "Invalid input num: expected==#{@option[:input_layers].length} actual==#{dataset.length}"
      end
      dataset_for_native = []
      dataset.each do |input|
        if !input[:data].instance_of?(Array) || input[:data].empty?
          raise "Invalid dataset for layer #{input[:name]}"
        end
        target_layer = @option[:input_layers].find { |item| item[:name] == input[:name] }
        expected_data_length = target_layer[:dims].inject(:*)
        if input[:data].length != expected_data_length
          raise "Invalid data length: expected==#{expected_data_length} actual==#{input[:data].length}"
        end
        dataset_for_native << input[:data]
      end

      # run
      results = native_run dataset_for_native

      # reshape result
      results.map do |raw|
        buffer = raw[:data]
        shape = raw[:shape]
        raw[:data] = transpose buffer, shape
      end

      yield results if block_given?
      results
    end
  end
end
