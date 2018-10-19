require 'mkmf'

dir_config('menoh')

if have_header("menoh/menoh.h") and have_library('menoh')
  create_makefile('menoh/menoh_native')
end
