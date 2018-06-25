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
      native_init menoh, option

      %i[batch_size channel_num width height].each do |key|
        raise "Required : #{key}" if option[key].nil?
        raise "Invalid option : #{key}" unless option[key].integer? && (option[key] > 0)
      end
      raise 'Required : input_layer' if option[:input_layer].nil?
      if !option[:input_layer].instance_of?(String) || option[:input_layer].empty?
        raise 'Invalid option : input_layer'
      end
      if option[:output_layers].nil? || option[:output_layers].empty?
        raise "Invalid ':output_layers'"
      end
      @option = option
      yield self if block_given?
    end

    def run(dataset)
      raise 'Invalid dataset' if !dataset.instance_of?(Array) || dataset.empty?
      expected_data_length = @option[:channel_num] * @option[:width] * @option[:height]
      dataset.each do |data|
        if data.length != expected_data_length
          raise "Invalid data length: expected==#{expected_data_length} actual==#{data.length}"
        end
      end

      # run
      raw_results = native_run dataset

      # reshape result
      raw_results.map do |raw|
        buffer = raw[:buffer]
        shape = raw[:shape]
        raw[:buffer] = transpose buffer, shape
      end
      results = []
      dataset.length.times do |i|
        result = {}
        raw_results.each do |raw|
          result[raw[:name]] = raw[:buffer][i]
        end
        results << result
      end
      yield results if block_given?
      results
    end
  end
end
