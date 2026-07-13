#
# pltlib — Fortran/C objects for xzplot
#
FC ?= gfortran
CC ?= gcc
FFLAGS ?= -O2 -std=legacy
CFLAGS ?= -O2 -Wall

# Debian/Ubuntu X11 headers; override if needed: make X11INC=/usr/include
X11INC ?= /usr/include/X11

.PHONY: all clean

all: pltsub.o xpltlib.o xbuplot.o

pltsub.o: pltsub.f
	$(FC) $(FFLAGS) -c $<

xpltlib.o: xpltlib.f
	$(FC) $(FFLAGS) -c $<

xbuplot.o: xbuplot.c
	$(CC) $(CFLAGS) -I$(X11INC) -c $<

clean:
	rm -f *.o
