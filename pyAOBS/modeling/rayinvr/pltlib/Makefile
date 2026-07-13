# 编译选项
FFLAGS = -O -fPIC

# 默认目标
all: pltsub.o pltlib.o

# 编译规则
.f.o:
	f77 $(FFLAGS) -c $<

# 清理规则
clean:
	rm -f *.o 