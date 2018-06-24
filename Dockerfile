FROM ubuntu:xenial-20180228

LABEL maintainer "Kunihiko Miyoshi <miyoshik@preferred.jp>"
LABEL OBJECT="Menoh Ruby Extension Reference Environment"

ENV RUNX_VERSION 0.4.0-alpha
ENV INSTALL_PREFIX /usr/local
ENV LD_LIBRARY_PATH ${INSTALL_PREFIX}/lib

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
    mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. && make && \
    make install

# Menoh
# TODO git clone
ADD external/Menoh-$RUNX_VERSION.zip /opt/
WORKDIR /opt/
RUN unzip Menoh-$RUNX_VERSION.zip && \
    cd Menoh-$RUNX_VERSION && \
    sed -i 's/add_subdirectory(example)//g' CMakeLists.txt && \
    sed -i 's/add_subdirectory(test)//g' CMakeLists.txt && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. && \
    make install

# menoh-ruby
RUN gem install rake-compiler
# RUN mkdir /opt/menoh-ruby
# ADD . /opt/menoh-ruby
# WORKDIR /opt/menoh-ruby
# RUN rake && rake install
