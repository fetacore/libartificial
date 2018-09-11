CC = gcc
DLLEXT = .so
SKIP_GPU := $(wildcard src/*/gpu_*.c)
SKIP_CPU := $(wildcard src/*/cpu_*.c)

RM = rm -f


SRCS := $(filter-out $(SKIP_GPU), $(wildcard src/*/*.c))
CFLAGS = -fPIC -Wall -Wextra -march=native -O3 -pedantic-errors
LDFLAGS = -shared
LIBS = -lm -lopenblas -lpthread
TARGET_LIB = libartificial$(DLLEXT)
OBJS = $(SRCS:.c=.o)


GPU_SRCS := $(filter-out $(SKIP_CPU), $(wildcard src/*/*.c))
GPU_CFLAGS = -L./clblast/build -Wl,-rpath=./clblast/build -fPIC -Wall -Wextra -march=native -O3 -pedantic-errors
GPU_LDFLAGS = -L./clblast/build -Wl,-rpath=./clblast/build -shared
GPU_LIBS = -lm -lclblast
GPU_TARGET_LIB = libartificialgpu$(DLLEXT)
GPU_OBJS = $(GPU_SRCS:.c=.o)

VALGRINDOPTS = valgrind --leak-check=yes --track-origins=yes

.PHONY: cpu
cpu: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^ $(LIBS) && make clean

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: gpu
gpu: ${GPU_TARGET_LIB}

$(GPU_TARGET_LIB): $(GPU_OBJS)
	$(CC) ${GPU_LDFLAGS} -o $@ $^ $(GPU_LIBS) && make clean

$(GPU_SRCS:.c=.d):%.d:%.c
	$(CC) $(GPU_CFLAGS) -MM $< >$@

include $(GPU_SRCS:.c=.d)

.PHONY: test1
test1:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -Wall -o ./test1 ./mlp_reg.c -lm -lartificial;\
	./test1

.PHONY: test2
test2:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -L../clblast/build -Wl,-rpath=../clblast/build -Wall -o ./test2 ./mlp_reg_gpu.c -lm -lclblast -lartificialgpu;\
	./test2

.PHONY: test3
test3:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -Wall -o ./test3 ./autoencoder.c -lm -lartificial;\
	./test3

.PHONY: test4
test4:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -Wall -o ./test4 ./cnn.c -lm -lartificial;\
	./test4

.PHONY: clean
clean:
	-${RM} ./src/*/*.o && $(RM) ./src/*/*.d
