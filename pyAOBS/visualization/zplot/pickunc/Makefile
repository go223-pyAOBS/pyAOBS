#
# makefile for PICKUNC 
#
EXEDIR=../bin/
#
# compile flags
#
FFLAGS =  -O 
#
#rules for compiling .f files:
#
.f.o:
	$(COMPILE.f) $(FFLAGS) $< $(INCS)
#---------------------------------------------------------------------------
#
# Source objects
#
PICKUNC_OBJS= main.o misc.o
#
# Executable name
#
PICKUNC_EXEC=$(EXEDIR)pickunc
#
pickunc: $(PICKUNC_EXEC)
$(PICKUNC_EXEC): ${PICKUNC_OBJS}
	f77 -o main ${PICKUNC_OBJS}
	mv main ${PICKUNC_EXEC}
#
# dependencies
#
main.o     :  zplot.par
