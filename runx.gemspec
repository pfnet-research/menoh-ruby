
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "runx/version"

Gem::Specification.new do |spec|
  spec.name          = "runx"
  spec.version       = Runx::VERSION
  spec.authors       = ["Kunihiko MIYOSHI"]
  spec.email         = ["miyoshik@preferred.jp"]

  spec.summary       = %q{Ruby binding of ONNX runtime engine 'Runx'}
  spec.description   = %q{Ruby binding of ONNX runtime engine 'Runx'}
  spec.homepage      = "https://github.com/colspan/runx-ruby"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/runx_native/extconf.rb"]

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "rake-compiler"
  spec.add_development_dependency "minitest", "~> 5.0"
end