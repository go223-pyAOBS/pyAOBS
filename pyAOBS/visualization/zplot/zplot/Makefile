#
# makefile for XZPLOT
#
EXEDIR=../bin/
PLTLIB=../pltlib
#
# compile flags
#
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
	f77 -o main ${XZPLOT_OBJS} -L/usr/X11R6/lib -lX11
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
	cd $(@D);cc -O -I/usr/X11R6/include/X11 -c xbuplot.c
FORCE:

