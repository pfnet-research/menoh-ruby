lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'menoh/version'

Gem::Specification.new do |spec|
  spec.name          = 'menoh'
  spec.version       = Menoh::VERSION
  spec.authors       = ['Kunihiko MIYOSHI']
  spec.email         = ['menoh-oss@preferred.jp']

  spec.summary       = "Ruby binding of ONNX runtime engine 'Menoh'"
  spec.description   = "Ruby binding of ONNX runtime engine 'Menoh'"
  spec.homepage      = 'https://github.com/pfnet-research/menoh-ruby'
  spec.license       = 'MIT'

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = 'exe'
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']
  spec.extensions    = ['ext/menoh_native/extconf.rb']

  spec.add_development_dependency 'bundler', '~> 1.16'
  spec.add_development_dependency 'minitest', '~> 5.0'
  spec.add_development_dependency 'pry'
  spec.add_development_dependency 'rake', '~> 10.0'
  spec.add_development_dependency 'rake-compiler'
  spec.add_development_dependency 'numo-narray'
end
