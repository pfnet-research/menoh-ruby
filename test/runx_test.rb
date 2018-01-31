require "test_helper"

class RunxTest < Minitest::Test
  def test_that_it_has_a_version_number
    refute_nil ::Runx::VERSION
  end
end
