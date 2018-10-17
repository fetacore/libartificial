CC = gcc
DLLEXT = .so
RM = rm -f

SRCS_CPU = $(filter-out $(wildcard src/*/gpu_*.c), $(wildcard src/*/*.c))
# SRCS_GPU = $(filter-out $(wildcard src/*/cpu_*.c), $(wildcard src/*/*.c))
SRCS = $(wildcard src/*/*.c)

OBJS_CPU = $(SRCS_CPU:.c=.o)
# OBJS_GPU = $(SRCS_GPU:.c=.o)
OBJS = $(SRCS:.c=.o)

TARGET_LIB_CPU = libartificial$(DLLEXT)
# TARGET_LIB_GPU = libartificial_gpu$(DLLEXT)
# TARGET_LIBS = $(TARGET_LIB_CPU) $(TARGET_LIB_GPU)

CFLAGS_CPU = -L./openblas -Wl,-rpath=./openblas -fPIC -Ofast -march=native -Wall -Wextra -Wshadow -Wpointer-arith -Wconversion -Wunreachable-code
# CFLAGS_GPU = -L./clblast/build -Wl,-rpath=./clblast/build -fopenmp -fPIC -ffast-math -flto -O3 -march=native -Wall

LDFLAGS_CPU = -L./openblas -Wl,-rpath=./openblas -shared
# LDFLAGS_GPU = -L./clblast/build -Wl,-rpath=./clblast/build -shared

LIBS_CPU = -lm -lopenblas
# LIBS_GPU = -lm -lclblast -lgomp

VALGRINDOPTS = valgrind --log-file="logfile" --leak-check=full --show-leak-kinds=all --track-origins=yes --dsymutil=yes --trace-children=yes -v

# .PHONY: all
# all: $(TARGET_LIBS)
# 
# $(TARGET_LIB_CPU): $(OBJS_CPU)
# 	$(CC) $(LDFLAGS_CPU) $(CFLAGS_CPU) -o $@ $^ $(LIBS_CPU) && make clean
# 	
# $(TARGET_LIB_GPU): $(OBJS_GPU)
# 	$(CC) $(LDFLAGS_GPU) $(CFLAGS_GPU) -o $@ $^ $(LIBS_GPU) && make clean

.PHONY: all
all: ${TARGET_LIB_CPU}

$(TARGET_LIB_CPU): $(OBJS_CPU)
	$(CC) ${LDFLAGS_CPU} $(CFLAGS_CPU) -o $@ $^ $(LIBS_CPU) && make clean

# .PHONY: gpu
# gpu: ${TARGET_LIB_GPU}
# 
# $(TARGET_LIB_GPU): $(OBJS_GPU)
# 	$(CC) ${LDFLAGS_GPU} $(CFLAGS_GPU) -o $@ $^ $(LIBS_GPU) && make clean

.PHONY: test1
test1:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -L../openblas -Wl,-rpath=../openblas -Wall -Ofast -o ./test1 ./mlp_reg.c -lm -lopenblas -lartificial;\
	time ./test1
	
# .PHONY: test2
# test2:
# 	make clean &&\
# 	cd ./examples/;\
# 	$(CC) -L../ -Wl,-rpath=../ -L../clblast/build -Wl,-rpath=../clblast/build -Wall -O3 -o ./test2 ./mlp_reg_gpu.c -lm -lclblast -lartificial_gpu;\
# 	time ./test2
	
.PHONY: test2
test2:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -L../openblas -Wl,-rpath=../openblas -Wall -Ofast -o ./test2 ./mlp_classification.c -lm -lopenblas -lartificial;\
	time ./test2

.PHONY: test3
test3:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -Wall -O3 -o ./test3 ./autoencoder.c -lm -lartificial;\
	./test3

.PHONY: test4
test5:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -Wall -O3 -o ./test4 ./cnn.c -lm -lartificial;\
	./test4

.PHONY: clean
clean:
	-${RM} ./src/*/*.o && $(RM) ./src/*/*.d
