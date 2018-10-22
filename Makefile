CC = gcc
VALGRINDOPTS = valgrind --log-file="logfile" --leak-check=full --show-leak-kinds=all --track-origins=yes --dsymutil=yes --trace-children=yes -v

# for multithreading
# -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math 

.PHONY: all
all: test1 test2 test3 test4

test1:
	cd ./examples/;\
	$(CC) -Ofast -march=native -Wall -o ./test1 ./mlp_reg.c -lm;

test2:
	cd ./examples/;\
	$(CC) -Ofast -march=native -Wall -o ./test2 ./mlp_classification.c -lm;
	
test3:
	cd ./examples/;\
	$(CC) -Ofast -march=native -Wall -o ./test3 ./autoencoder.c -lm;

test4:
	cd ./examples/;\
	$(CC) -Ofast -march=native -Wall -o ./test4 ./cnn.c -lm;

.PHONY: test1
test1:
	cd ./examples/;\
	$(CC) -Ofast -march=native -Wall -o ./test1 ./mlp_reg.c -lm;
	
.PHONY: test2
test2:
	cd ./examples/;\
	$(CC) -Ofast -march=native -Wall -o ./test2 ./mlp_classification.c -lm;

.PHONY: test3
test3:
	cd ./examples/;\
	$(CC) -Ofast -march=native -Wall -o ./test3 ./autoencoder.c -lm;
	
.PHONY: test4
test4:
	cd ./examples/;\
	$(CC) -Ofast -march=native -Wall -o ./test4 ./cnn.c -lm;
