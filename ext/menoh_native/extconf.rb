require 'mkmf'

# have_library("stdc++")
have_library('mkldnn')
have_library('protobuf')

menoh_dir = dir_config('menoh')
$INCFLAGS << " -I#{menoh_dir[0]}/menoh"
have_library('menoh')

# $CPPFLAGS << " -std=c++14"
$DLDFLAGS << ' -rdynamic'

create_makefile('menoh/menoh_native')
