# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure('2') do |config|
  config.vm.box = 'ubuntu/xenial64'

  config.vm.provider 'virtualbox' do |vb|
    vb.memory = '4096'
    vb.cpus = 2
  end

  config.vm.provision 'shell', inline: <<-SHELL
    export INSTALL_PREFIX=/usr/local
    sudo apt update
    sudo apt install -y gcc g++ cmake cmake-data libopencv-dev
    sudo apt install -y ruby-dev ruby-rmagick
    sudo gem install bundler rake-compiler

    # protobuf
    sudo apt install -y protobuf-compiler libprotobuf-dev

    # mkl-dnn
    cd
    git clone https://github.com/01org/mkl-dnn.git
    cd mkl-dnn/scripts && bash ./prepare_mkl.sh && cd ..
    sed -i 's/add_subdirectory(examples)//g' CMakeLists.txt
    sed -i 's/add_subdirectory(tests)//g' CMakeLists.txt
    mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. && make
    sudo make install

    # Menoh
    cd
    git clone https://github.com/pfnet-research/menoh.git
    cd menoh
    sed -i 's/add_subdirectory(example)//g' CMakeLists.txt
    sed -i 's/add_subdirectory(test)//g' CMakeLists.txt
    mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. && make
    sudo make install

    # menoh-ruby
    export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:$LD_LIBRARY_PATH
    cd /vagrant
    rake
    sudo rake install
  SHELL
end
