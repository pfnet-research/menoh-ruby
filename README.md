# Menoh Ruby Extension 

This is a Ruby extension of [Menoh](https://github.com/pfnet-research/menoh); an ONNX runtime engine developed by [@okdshin](https://github.com/okdshin) and their team [@pfnet-research](https://github.com/pfnet-research).

## Installation

You need `ruby-dev`, `bundler`, `rake-compiler` to install this extension.

```bash
$ sudo apt install ruby-dev
$ sudo gem install bundler
$ sudo gem install rake-compiler
```

And add this line to your application's Gemfile:

```ruby
gem 'menoh'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install menoh

## Usage

Please see [Menoh tutorial](https://github.com/pfnet-research/menoh/blob/master/docs/tutorial.md) and [menoh-ruby tutorial](https://github.com/pfnet-research/menoh-ruby/blob/master/docs/tutorial.md).
And we have [some examples](https://github.com/pfnet-research/menoh/blob/master/example/) on this repository.

## Development

After checking out the repo, run `bin/setup` to install ruby dependencies. Then, run `rake test` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

### Docker

You can develop on docker. For details, please refer to [Dockerfile](Dockerfile).

```bash
$ export IMAGE_VERSION=0.0.0 # please specify version
$ sudo -E docker build -t menoh-ruby:$IMAGE_VERSION `pwd`
$ sudo -E docker run -it --name menoh-ruby-test -v $(pwd):/opt/menoh-ruby --entrypoint /bin/bash menoh-ruby:$IMAGE_VERSION
＄ cd /opt/menoh-ruby
＄ rake && rake install

```

#### attach after stop

```bash
sudo docker start menoh-ruby-test bash
sudo docker attach menoh-ruby-test
```

### Vagrant

You can also set up the development environment by using Vagrant. The details are available on [Vagrantfile](Vagrantfile).

```bash
$ vagrant up
$ vagrant ssh
$ cd /vagrant
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/colspan/menoh-ruby. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the OnnxInstant project’s codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/colspan/menoh-ruby/blob/master/CODE_OF_CONDUCT.md).
