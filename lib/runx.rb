require "runx/version"
require "runx/runx_native"

module Runx
  class Runx
    def initialize file
      if not file.instance_of?(String) or not File.exist?(file)
        raise "No such file : #{file}"
      end

      native_init file
      if block_given?
        begin
          yield self
        ensure
          # do nothing
        end
      end
    end
    def make_model condition
      if condition[:output_layers] == nil or condition[:output_layers].length == 0
        raise "Invalid ':output_layers'" 
      end
      # TODO no such layer
      if condition[:backend] == nil or condition[:backend] != "mkldnn"
        raise "Invalid ':backend' : #{condition[:backend]}"
      end
      RunxModel.new self, condition
    end
  end
end

module Runx
  class RunxModel
    def initialize runx, condition
      # TODO check condition
      native_init runx, condition
      if block_given?
        begin
          yield self
        ensure
          # do nothing
        end
      end
    end
    def run dataset, condition
      if not dataset.instance_of?(Array) or dataset.length == 0
        raise "Invalid dataset" 
      end
      [:channel_num, :width, :height].each do |key|
        if condition[key] == nil
          raise "Required : #{key.to_s}"
        end
        if not condition[key].integer?
          raise "Invalid option : #{key.to_s}" 
        end
      end
      if condition[:input_layer] == nil
        raise "Required : #{:input_layer.to_s}"
      end
      if not condition[:input_layer].instance_of?(String) or condition[:input_layer].length == 0
        raise "Invalid option : #{:input_layer.to_s}" 
      end

      # TODO no such layer      
      expected_data_length = condition[:channel_num] * condition[:width] * condition[:height]
      dataset.each do |data|
        if data.length != expected_data_length
          raise "Invalid data length: expected==#{expected_data_length} actual==#{data.length}"
        end
      end
      a = native_run dataset, condition
      p a
    end
  end
end