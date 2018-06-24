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

    def make_model(condition)
      if condition[:backend].nil? || (condition[:backend] != 'mkldnn')
        raise "Invalid ':backend' : #{condition[:backend]}"
      end
      model = MenohModel.new self, condition
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
    def initialize(menoh, condition)
      native_init menoh, condition
      yield self if block_given?
    end

    def run(dataset, condition)
      raise 'Invalid dataset' if !dataset.instance_of?(Array) || dataset.empty?
      %i[channel_num width height].each do |key|
        raise "Required : #{key}" if condition[key].nil?
        raise "Invalid option : #{key}" unless condition[key].integer?
      end
      raise 'Required : input_layer' if condition[:input_layer].nil?
      if !condition[:input_layer].instance_of?(String) || condition[:input_layer].empty?
        raise 'Invalid option : input_layer'
      end
      if condition[:output_layers].nil? || condition[:output_layers].empty?
        raise "Invalid ':output_layers'"
      end

      expected_data_length = condition[:channel_num] * condition[:width] * condition[:height]
      dataset.each do |data|
        if data.length != expected_data_length
          raise "Invalid data length: expected==#{expected_data_length} actual==#{data.length}"
        end
      end

      # run
      raw_results = native_run dataset, condition

      # reshape result
      raw_results.map do |raw|
        buffer = raw[:buffer]
        shape = raw[:shape]
        raw[:buffer] = transpose buffer, shape
      end
      results = []
      output_layers = raw_results.map { |result| result[:name] }
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
