FROM ubuntu:xenial-20180228

MAINTAINER Kunihiko Miyoshi
LABEL OBJECT="Runx Ruby Extension Reference Environment"

ENV RUNX_VERSION 0.2.1-alpha

RUN apt-get update && apt-get install -y \
  git \
  gcc \
  g++ \
  cmake \
  cmake-data \
  libopencv-dev \
  protobuf-compiler \
  libprotobuf-dev \
  ruby-dev \
  ruby-rmagick \
  ruby-bundler && \
  rm -rf /var/lib/apt/lists/*

# MKL-DNN
RUN mkdir /opt/mkl-dnn
WORKDIR /opt/mkl-dnn
RUN git clone https://github.com/01org/mkl-dnn.git && \
    cd mkl-dnn/scripts && bash ./prepare_mkl.sh && cd .. && \
    sed -i 's/add_subdirectory(examples)//g' CMakeLists.txt && \
    sed -i 's/add_subdirectory(tests)//g' CMakeLists.txt && \
    mkdir -p build && cd build && cmake .. && make && \
    make install

# Runx
# TODO git clone
ADD external/Runx-$RUNX_VERSION.zip /opt/
WORKDIR /opt/
RUN unzip Runx-$RUNX_VERSION.zip && \
    cd Runx-$RUNX_VERSION && \
    sed -i 's/add_subdirectory(test)//g' CMakeLists.txt && \
    sed -i 's/add_subdirectory(example)//g' CMakeLists.txt && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install

# runx-ruby
RUN gem install rake-compiler
RUN mkdir /opt/runx-ruby
ADD . /opt/runx-ruby
WORKDIR /opt/runx-ruby
RUN rake && bundle
