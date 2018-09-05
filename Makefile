DLLEXT = .so

SRCS = $(wildcard src/*/*.c)

CC = gcc
CFLAGS = -fPIC -Wall -Wextra -march=native -O3 -g -pedantic-errors
LDFLAGS = -shared
RM = rm -f
TARGET_LIB = libartificial$(DLLEXT)
LIBS = -lm -lopenblas -lpthread -lgfortran

VALGRINDOPTS = valgrind --leak-check=yes --track-origins=yes

OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^ $(LIBS) && make clean

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

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
	$(CC) -L../ -Wl,-rpath=../ -Wall -o ./test2 ./autoencoder.c -lm -lartificial;\
	./test2

.PHONY: test3
test3:
	make clean &&\
	cd ./examples/;\
	$(CC) -L../ -Wl,-rpath=../ -Wall -o ./test3 ./cnn.c -lm -lartificial;\
	./test3

.PHONY: clean
clean:
	-${RM} ${OBJS} $(SRCS:.c=.d)

.PHONY: cleanall
cleanall:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d)
	
