#     File patched for g77 compilation
#     Scott Pearse / email: scott.pearse@gmail.com / web: http://www.linkedin.com/in/scottpearse
#
#
# makefile for RAYINVR
#
EXEDIR=./bin/
PLTLIB=./pltlib/

# 设置编译器
FC = gfortran
#
# compile flags
#
FFLAGS = -O -fPIC  # 添加-fPIC选项用于生成位置无关代码
CFLAGS = -O
#
#rules for compiling .f files:
#
.f.o:
	$(FC) $(FFLAGS) -c $<
.c.o:
	$(COMPILE.c) $(CFLAGS) $< $(INCS)
#---------------------------------------------------------------------------
#
# Source objects
#
RAYINVR_OBJS= main.o trc.o atrc.o adjpt.o rngkta.o plt.o misc.o inv.o hdw.o calmod.o blkdat.o $(PLTLIB)/pltsub.o $(PLTLIB)/pltlib.o
XRAYINVR_OBJS= main.o trc.o atrc.o adjpt.o rngkta.o plt.o misc.o inv.o hdw.o calmod.o blkdat.o $(PLTLIB)/pltsub.o $(PLTLIB)/xpltlib.o $(PLTLIB)/xbuplot.o
SHARED_OBJS= rayinvr_interface.o main.o ini_setup.o trc.o atrc.o adjpt.o rngkta.o plt.o misc.o inv.o hdw.o calmod.o blkdat.o $(PLTLIB)/pltsub.o $(PLTLIB)/pltlib.o

# 添加共享库目标
RAYINVR_LIB=librayinvr.so

# 默认目标
all: $(RAYINVR_EXEC) $(XRAYINVR_EXEC) $(RAYINVR_LIB)

# Executable name
RAYINVR_EXEC=$(EXEDIR)rayinvr 
XRAYINVR_EXEC=$(EXEDIR)xrayinvr

rayinvr: $(RAYINVR_EXEC)
$(RAYINVR_EXEC): ${RAYINVR_OBJS}
	$(FC) -o main ${RAYINVR_OBJS}
	mv main ${RAYINVR_EXEC}
#	unilink ${RAYINVR_OBJS}
#
xrayinvr: $(XRAYINVR_EXEC)
$(XRAYINVR_EXEC): ${XRAYINVR_OBJS}
	$(FC) -o main ${XRAYINVR_OBJS} -L/lib/x86_64-linux-gnu -lX11
	mv main ${XRAYINVR_EXEC}
#
# dependencies
#
main.o     :  rayinvr.par rayinvr.com
trc.o      :  rayinvr.par rayinvr.com
atrc.o     :  rayinvr.par rayinvr.com
adjpt.o    :  rayinvr.par rayinvr.com
plt.o      :  rayinvr.par rayinvr.com
misc.o     :  rayinvr.par rayinvr.com
inv.o      :  rayinvr.par rayinvr.com
blkdat.o   :  rayinvr.par rayinvr.com
hdw.o      :  rayinvr.par rayinvr.com
calmod.o   :  rayinvr.par rayinvr.com
rngkta.o   :  rayinvr.par rayinvr.com
rayinvr_interface.o: rayinvr.par rayinvr.com rayinvr_interface.com
 
$(PLTLIB)/pltsub.o: FORCE
	cd $(@D); $(FC) -O -fPIC -c pltsub.f
FORCE:
 
$(PLTLIB)/pltlib.o: FORCE
	cd $(@D); $(FC) -O -fPIC -c pltlib.f
FORCE:
 
$(PLTLIB)/xpltlib.o: FORCE
	cd $(@D); $(MAKE) $(@F)
FORCE:
 

$(PLTLIB)/xbuplot.o: FORCE
	cd $(@D);cc -O -I/usr/include/X11 -c xbuplot.c
FORCE:

# 添加共享库编译规则
$(RAYINVR_LIB): $(SHARED_OBJS)
	$(FC) -shared -o $@ $(SHARED_OBJS)

# 清理规则
clean:
	rm -f *.o *.mod *.so $(RAYINVR_EXEC) $(XRAYINVR_EXEC)

