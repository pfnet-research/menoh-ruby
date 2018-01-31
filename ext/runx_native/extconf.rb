require "mkmf"

have_library("stdc++")
have_library("mkldnn")
have_library("protobuf")

runx_dir = dir_config('runx')
runx_dir = dir_config('runx', '/home/kuni/local/opt/include', '/home/kuni/local/opt/lib')
$INCFLAGS << " -I#{runx_dir[0]}/runx"
have_library("runx")

$CPPFLAGS << " -std=c++14"
$DLDFLAGS << " -rdynamic"

create_makefile("runx/runx_native")
