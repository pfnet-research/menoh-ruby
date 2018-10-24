require 'mkmf'

if pkg_config("menoh")
  create_makefile('menoh/menoh_native')
end
