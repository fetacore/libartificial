# libartificial

This is a pure C shared library for arbitrarily deep neural networks. It is my first attempt to write a scientific project in C but the speed is already outstanding. I made this library in order to assist me with my PhD Thesis research questions.

It is CPU ([OpenBLAS](https://github.com/xianyi/OpenBLAS)) and GPU ([CLBlast](https://github.com/CNugteren/CLBlast)) friendly (soon with support for cuBLAS if I get my hands on an NVIDIA GPU). I have plans to extend it for CNNs and RNNs. The bindings for Python and JS (with webassembly) will be ready soon.

The feedforward procedure does not have a hardcoded depth (it can have as many layers as you want).

## Getting Started

The library is created with Linux machines in mind (OSX users should just need to change the compilation command from gcc to clang and change the corresponding flags).

In order to get libartificial you have to do the following (assuming working installation of git)

```
git clone https://github.com/jroukis/libartificial.git
cd libartificial
rm -rf .git
```

### Prerequisites

In order to compile the library for CPU you need to install [OpenBLAS](https://github.com/xianyi/OpenBLAS).
In order to compile the library for GPU you need to install [CLBlast](https://github.com/CNugteren/CLBlast).

## Specifics for CLBlast

It is recommended to do the optimizations proposed by the author. The library assumes that CLBlast is in the libartificial folder under the name "clblast".
You do the following:

```
git clone https://github.com/CNugteren/CLBlast.git clblast
cd clblast && mkdir build && cd build
cmake .. && make
cd ../../

```

You should also take note that the library uses doubles and not all GPUs support double arithmetic in OpenCL.

### Compilation

In order to compile the library do the following (assuming you continue from where we left off)

- For CPU

```
make cpu
```
- For GPU

```
make gpu
```

## Examples

For the time being I have four examples which you can find in the "examples" folder:
- MLP regression with CPU:

```
make test1
```

- MLP regression with GPU

```
make test2
```

- Autoencoder (CPU):

```
make test3
```

- CNN (only im2col at the moment):

```
make test4
```

## API

This part is being written at the moment

## Donations

If you like my work and/or you want to use it for your own projects or want me to create a custom recipe for you, I would gladly accept your donations at:

BTC: 1HzxXZPQSNg7U53XoBSWCpugKUg5DaZELu

ETH: 0xf09fce52f7ecd940cae2826deae151b6495354f6

## License

Copyright (c) Jim Karoukis.
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
