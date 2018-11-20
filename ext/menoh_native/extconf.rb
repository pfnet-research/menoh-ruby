require 'mkmf'

if try_cflags '-std=gnu11'
  $CFLAGS += " -std=gnu11"
elsif try_cflags '-std=c11'
  $CFLAGS += " -std=c11"
elsif try_cflags '-std=gnu99'
  $CFLAGS += " -std=gnu99"
elsif try_cflags '-std=c99'
  $CFLAGS += " -std=c99"
end

if pkg_config("menoh")
  have_const('menoh_dtype_float64', 'menoh/menoh.h')
  create_makefile('menoh/menoh_native')
end
