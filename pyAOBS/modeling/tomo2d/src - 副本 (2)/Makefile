## TOMO2D Makefile (portable + OpenMP ready)

CXX ?= g++
CC  ?= gcc

# Build options:
#   make                       # release build
#   make USE_OPENMP=0          # disable OpenMP
#   make CXX=g++-13            # custom compiler
USE_OPENMP ?= 1
QUIET_LEGACY_WARN ?= 1

CXXFLAGS ?= -O3 -std=gnu++17 -Wall -Wextra -Wno-unused-parameter
CPPFLAGS ?= -I.
LDFLAGS  ?=
LDLIBS   ?= -lm

ifeq ($(USE_OPENMP),1)
  CXXFLAGS += -fopenmp
  LDFLAGS  += -fopenmp
endif

# Legacy Numerical Recipes style .c/.h compiled as C++ emits many cosmetic warnings.
ifeq ($(QUIET_LEGACY_WARN),1)
  CXXFLAGS += -Wno-write-strings -Wno-sign-compare -Wno-reorder -Wno-maybe-uninitialized \
              -Wno-unused-variable -Wno-unused-local-typedefs -Wno-unused-value
endif

SRC_GEN_SMESH = gen_smesh.cc zeltform.cc geom.cc error.cc util.cc
OBJ_GEN_SMESH = $(SRC_GEN_SMESH:.cc=.o)

SRC_GEN_DAMP = gen_damp.cc zeltform.cc geom.cc error.cc util.cc
OBJ_GEN_DAMP = $(SRC_GEN_DAMP:.cc=.o)

SRC_GEN_VCORR = gen_vcorr.cc zeltform.cc geom.cc error.cc util.cc
OBJ_GEN_VCORR = $(SRC_GEN_VCORR:.cc=.o)

SRC_EDIT_SMESH_HHB = edit_smesh_HHB.cc geom.cc error.cc util.cc smesh.cc interface.cc corrlen.cc
SRC_EDIT_SMESH_HHB_C = d_jacobi.c d_choldc.c nrutil.c
OBJ_EDIT_SMESH_HHB = $(SRC_EDIT_SMESH_HHB:.cc=.o) $(SRC_EDIT_SMESH_HHB_C:.c=.o)

SRC_STAT_SMESH = stat_smesh.cc error.cc interface.cc util.cc
OBJ_STAT_SMESH = $(SRC_STAT_SMESH:.cc=.o)

SRC_TT_FORWARD = tt_forward.cc syngen.cc smesh.cc graph.cc \
                 traveltime.cc betaspline.cc bend.cc bend_mnbrak.cc bend_brent.cc \
                 interface.cc corrlen.cc geom.cc error.cc util.cc
SRC_TT_FORWARD_C = d_jacobi.c d_choldc.c nrutil.c
OBJ_TT_FORWARD = $(SRC_TT_FORWARD:.cc=.o) $(SRC_TT_FORWARD_C:.c=.o)

SRC_TT_INVERSE = tt_inverse.cc inverse.cc sparse_rect.cc lsqr.cc \
                 smesh.cc graph.cc traveltime.cc betaspline.cc \
                 bend.cc bend_mnbrak.cc bend_brent.cc \
                 interface.cc corrlen.cc geom.cc error.cc util.cc jgrav.cc
SRC_TT_INVERSE_C = d_jacobi.c d_choldc.c nrutil.c d_realft.c d_four1.c
OBJ_TT_INVERSE = $(SRC_TT_INVERSE:.cc=.o) $(SRC_TT_INVERSE_C:.c=.o)

# Important: compile legacy .c numerics with C++ to keep symbols compatible
%.o: %.c
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x c++ -c $< -o $@

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: all clean
all: gen_smesh gen_damp gen_vcorr edit_smesh_HHB stat_smesh tt_forward tt_inverse

gen_smesh: $(OBJ_GEN_SMESH)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ_GEN_SMESH) $(LDLIBS)

gen_damp: $(OBJ_GEN_DAMP)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ_GEN_DAMP) $(LDLIBS)

gen_vcorr: $(OBJ_GEN_VCORR)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ_GEN_VCORR) $(LDLIBS)

edit_smesh_HHB: $(OBJ_EDIT_SMESH_HHB)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ_EDIT_SMESH_HHB) $(LDLIBS)

stat_smesh: $(OBJ_STAT_SMESH)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ_STAT_SMESH) $(LDLIBS)

tt_forward: $(OBJ_TT_FORWARD)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ_TT_FORWARD) $(LDLIBS)

tt_inverse: $(OBJ_TT_INVERSE)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ_TT_INVERSE) $(LDLIBS)

clean:
	rm -f *.o gen_smesh gen_damp gen_vcorr edit_smesh_HHB stat_smesh tt_forward tt_inverse

