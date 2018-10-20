CC = gcc
VALGRINDOPTS = valgrind --log-file="logfile" --leak-check=full --show-leak-kinds=all --track-origins=yes --dsymutil=yes --trace-children=yes -v

.PHONY: all
all: test1 test2 test3 test4

test1:
	cd ./examples/;\
	$(CC) -L../OpenBLAS -Wl,-rpath=../OpenBLAS -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math -Ofast -march=native -Wall -o ./test1 ./mlp_reg.c -lm -lopenblas;

test2:
	cd ./examples/;\
	$(CC) -L../OpenBLAS -Wl,-rpath=../OpenBLAS -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math -Ofast -march=native -Wall -o ./test2 ./mlp_classification.c -lm -lopenblas;
	
test3:
	cd ./examples/;\
	$(CC) -L../OpenBLAS -Wl,-rpath=../OpenBLAS -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math -Ofast -march=native -Wall -o ./test3 ./autoencoder.c -lm -lopenblas;

test4:
	cd ./examples/;\
	$(CC) -L../OpenBLAS -Wl,-rpath=../OpenBLAS -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math -Ofast -march=native -Wall -o ./test4 ./cnn.c -lm -lopenblas;

.PHONY: test1
test1:
	cd ./examples/;\
	$(CC) -L../OpenBLAS -Wl,-rpath=../OpenBLAS -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math -Ofast -march=native -Wall -o ./test1 ./mlp_reg.c -lm -lopenblas;
	
.PHONY: test2
test2:
	cd ./examples/;\
	$(CC) -L../OpenBLAS -Wl,-rpath=../OpenBLAS -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math -Ofast -march=native -Wall -o ./test2 ./mlp_classification.c -lm -lopenblas;

.PHONY: test3
test3:
	cd ./examples/;\
	$(CC) -L../OpenBLAS -Wl,-rpath=../OpenBLAS -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math -Ofast -march=native -Wall -o ./test3 ./autoencoder.c -lm -lopenblas;
	
.PHONY: test4
test4:
	cd ./examples/;\
	$(CC) -L../OpenBLAS -Wl,-rpath=../OpenBLAS -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math -Ofast -march=native -Wall -o ./test4 ./cnn.c -lm -lopenblas;
