CC = gcc
DLLEXT = .so
RM = rm -f

SRCS_CPU = $(filter-out $(wildcard src/*/gpu_*.c), $(wildcard src/*/*.c))
SRCS_GPU = $(filter-out $(wildcard src/*/cpu_*.c), $(wildcard src/*/*.c))
SRCS = $(wildcard src/*/*.c)

OBJS_CPU = $(SRCS_CPU:.c=.o)
OBJS_GPU = $(SRCS_GPU:.c=.o)
OBJS = $(OBJS_CPU) $(OBJS_GPU)

TARGET_LIB_CPU = libartificial$(DLLEXT)
TARGET_LIB_GPU = libartificial_gpu$(DLLEXT)
TARGET_LIBS = $(TARGET_LIB_CPU) $(TARGET_LIB_GPU)

CFLAGS_CPU = -fPIC -pthread -Wall -Wextra -march=native -O3 -pedantic-errors
CFLAGS_GPU = -L./clblast/build -Wl,-rpath=./clblast/build -fPIC -pthread -Wall -Wextra -march=native -O3 -pedantic-errors

LDFLAGS_CPU = -shared
LDFLAGS_GPU = -L./clblast/build -Wl,-rpath=./clblast/build -shared

LIBS_CPU = -lm -lopenblas -lpthread
LIBS_GPU = -lm -lclblast -lpthread

VALGRINDOPTS = valgrind --leak-check=yes --track-origins=yes

.PHONY: all
all: $(TARGET_LIBS)

$(TARGET_LIB_CPU): $(OBJS_CPU)
	$(CC) $(LDFLAGS_CPU) $(CFLAGS_CPU) -o $@ $^ $(LIBS_CPU) && make clean

$(TARGET_LIB_GPU): $(OBJS_GPU)
	$(CC) $(LDFLAGS_GPU) $(CFLAGS_GPU) -o $@ $^ $(LIBS_GPU) && make clean

.PHONY: cpu
cpu: ${TARGET_LIB_CPU}

$(TARGET_LIB_CPU): $(OBJS_CPU)
	$(CC) ${LDFLAGS_CPU} $(CFLAGS_CPU) -o $@ $^ $(LIBS_CPU) && make clean

.PHONY: gpu
gpu: ${TARGET_LIB_GPU}

$(TARGET_LIB_GPU): $(OBJS_GPU)
	$(CC) ${LDFLAGS_GPU} $(CFLAGS_GPU) -o $@ $^ $(LIBS_GPU) && make clean

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
	$(CC) -L../ -Wl,-rpath=../ -L../clblast/build -Wl,-rpath=../clblast/build -Wall -o ./test2 ./mlp_reg_gpu.c -lm -lclblast -lartificial_gpu;\
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
