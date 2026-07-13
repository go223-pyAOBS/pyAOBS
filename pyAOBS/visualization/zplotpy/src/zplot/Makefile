#
# makefile for XZPLOT
#
EXEDIR=../bin/
PLTLIB=../pltlib
#
# Use gfortran/gcc on Linux (override with make FC=... CC=...)
#
FC ?= gfortran
CC ?= gcc
#
# compile flags
#
FFLAGS = -O2 -std=legacy
#CFLAGS =  -O -I/usr/openwin/include
CFLAGS = -O2 -Wall
#
# X11: old Makefile used /usr/X11R6; typical Linux paths below (override if needed)
#
X11INC ?= /usr/include/X11
X11LIBDIR ?= /usr/lib/x86_64-linux-gnu
#
#rules for compiling .f files and .c files:
#
.f.o:
	$(FC) -c $(FFLAGS) $< $(INCS)
.c.o:
	$(CC) -c $(CFLAGS) $< $(INCS)
#---------------------------------------------------------------------------
#
# Source objects
#
XZPLOT_OBJS= main.o readf.o bandpass.o misc.o shade.o realft.o $(PLTLIB)/pltsub.o $(PLTLIB)/xpltlib.o $(PLTLIB)/xbuplot.o
#
# Executable name
#
XZPLOT_EXEC=$(EXEDIR)xzplot
#
xzplot: $(XZPLOT_EXEC)
$(XZPLOT_EXEC): ${XZPLOT_OBJS}
	$(FC) -o main ${XZPLOT_OBJS} -L$(X11LIBDIR) -lX11
	mv main ${XZPLOT_EXEC}
#
# dependencies
#
main.o     :  zplot.par
readf.o    :  zplot.par
misc.o     :  zplot.par
shade.o    :  zplot.par

$(PLTLIB)/pltsub.o: FORCE
	cd $(@D); $(MAKE) $(@F)
FORCE:

$(PLTLIB)/xpltlib.o: FORCE
	cd $(@D); $(MAKE) $(@F)
FORCE:

$(PLTLIB)/xbuplot.o: FORCE
	cd $(@D); $(MAKE) xbuplot.o CC="$(CC)" CFLAGS="$(CFLAGS)" X11INC="$(X11INC)"
FORCE:

.PHONY: clean
clean:
	rm -f *.o main
	$(MAKE) -C $(PLTLIB) clean

