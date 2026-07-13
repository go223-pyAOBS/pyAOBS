FFLAGS =  -O
#CFLAGS =  -O -I/usr/openwin/include
CFLAGS =  -O
#
#rules for compiling .f files and .c files:
#
.f.o:
	$(COMPILE.f) $(FFLAGS) $< $(INCS)
.c.o:
	$(COMPILE.c) $(CFLAGS) $< $(INCS)
#--------------------------------------------------------------------------------
SGY2Z_OBJS= screen.o fileio.o out_segy.o conv.o conv_intf.o distance.o geocar.o dump.o
sgy2z:  ${SGY2Z_OBJS}
	f77 -o ../bin/sgy2z sgy2z.f ${SGY2Z_OBJS}

FORCE:

screen.o:
	f77 -c screen.f

fileio.o:
	cc -c fileio.c

out_segy.o:
	cc -c out_segy.c

conv.o:
	cc -c conv.c

conv_intf.o:
	cc -c conv_intf.c

distance.o:
	f77 -c distance.f

geocar.o:
	f77 -c geocar.f

dump.o:
	f77 -c dump.f
