require "test_helper"

class OnnxInstantTest < Minitest::Test
  def test_that_it_has_a_version_number
    refute_nil ::OnnxInstant::VERSION
  end
end
