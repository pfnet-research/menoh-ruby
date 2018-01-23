require "mkmf"

have_library("stdc++")
have_library("mkldnn")
have_library("protobuf")

instant_dir = dir_config('instant')
# instant_dir = dir_config('instant', '/home/kuni/local/opt/instant/include', '/home/kuni/local/opt/instant/lib')
$INCFLAGS << " -I#{instant_dir[0]}/instant"
have_library("instant")

$CPPFLAGS << " -std=c++14"
$DLDFLAGS << " -rdynamic"

create_makefile("onnx_instant/onnx_instant")
