# libartificial

This is a small pure C header-only library for arbitrarily deep neural networks. It is my first attempt to write a scientific project in C but the speed is already outstanding. I made this library in order to assist me with my doctoral research.

It is CPU only (soon with support for cuBLAS if I get my hands on an NVIDIA GPU). I have plans to extend it for CNNs.

## Bindings

- Python: [python-libartificial](https://github.com/fetacore/python-libartificial)
- Javascript (WASM): soon
- R: soon

The procedure does not have a hardcoded depth (it can have as many layers with as many nodes as you want-beware of your heap limits).

## Getting Started

The library is created with Linux machines in mind but OSX users should not have a problem if they have gcc installed.
I will try to compile an example with Visual Studio and get back to you on how to do it.

In order to get libartificial you have to do the following (assuming working installation of git)

```
git clone --recurse-submodules -j8 https://github.com/fetacore/libartificial.git
cd libartificial
rm -rf .git
```

### Prerequisites

In order to compile the examples (and any other program using the header) for CPU you need to build [OpenBLAS](https://github.com/xianyi/OpenBLAS).
In order to check the procedure for your machine go to the project's [wiki](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide).

## Examples

For the time being I have four examples which you can find in the "examples" folder (assuming make and gcc):
- MLP regression (rbf like: gaussian & linear w/ RMSE):

```
make test1
```

- MLP classification (logistic & softmax w/ cross-entropy):

```
make test2
```

- Autoencoder:

```
make test3
```

- CNN (only im2col at the moment):

```
make test4
```
If you want to compile them all then just do

```
make
```

## Donations

If you like my work and/or you want to use it for your own projects or want me to create a custom recipe for you, I would gladly accept your donations at:

BTC: 1HzxXZPQSNg7U53XoBSWCpugKUg5DaZELu

ETH: 0xf09fce52f7ecd940cae2826deae151b6495354f6

## License

Copyright (c) Jim Karoukis.
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
