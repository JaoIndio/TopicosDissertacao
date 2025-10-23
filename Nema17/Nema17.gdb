# gdb -x startup.gdb

set confirm off
target remote localhost:3333
exec-file Debug/Nema17.out
symbol-file Debug/Nema17.out
load
monitor reset halt
break tm4c123gh6pm_startup_ccs.c:289
