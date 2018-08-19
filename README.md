# libartificial

This is a pure C shared library for arbitrarily deep neural networks. It is my first attempt to write a scientific project in C but the speed of the tests is already outstanding compared to the implementation in Python. I made this library in order to assist me with my PhD Thesis research questions.

It is CPU-only at the moment but I have plans to extend it with [CLBlast](https://github.com/CNugteren/CLBlast) or cuBLAS (if I get my hands on an NVIDIA GPU). I also have plans to extend it for CNNs and RNNs. The bindings for Python and JS (with webassembly) will be ready soon.

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

In order to compile the library you need to install [OpenBLAS](https://github.com/xianyi/OpenBLAS) and, in case your OS does not have it, libgfortran.

### Compilation

In order to compile the library do the following (assuming you continue from where we left off)

```
make
```

## Examples

For the time being I have two examples which you can find in the "examples" folder:
- MLP regression that you can run as follows:
```
make test1
```

- Autoencoder:
```
make test2
```

## API

This part is being written at the moment

## License

Copyright (c) Jim Karoukis.
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
