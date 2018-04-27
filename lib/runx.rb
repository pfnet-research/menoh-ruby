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
      if not condition[:input_layer].instance_of?(String)
        raise "Invalid ':input_layer'" 
      end
      # TODO no such layer      
      if not condition[:channel_num].instance_of?(Fixnum)
        raise "Invalid ':channel_num'" 
      end
      if not condition[:width].instance_of?(Fixnum)
        raise "Invalid ':width'" 
      end
      if not condition[:height].instance_of?(Fixnum)
        raise "Invalid ':height'" 
      end
      native_run dataset, condition
    end
  end
end