# Surface-Only Dynamic Deformables using a Boundary Element Method

This repository is the official implementation of [Surface-Only Dynamic Deformables using a Boundary Element Method](https://rsugimoto.net/SurfaceOnlyDynamicDeformablesProject) with a few extensions.

## Implemented Core Simulators

- Elastodynamics Convolution Quadrature BEM (CQBEM) [[Schanz and Antes 1997](https://doi.org/10.1007/s004660050265)]

  > The default BEM method for elastodynamics with high stability. Discussed fully in our paper.

- Elastodynamics Dual Reciprocity BEM (DRBEM) [[Agnantiaris et al. 1998](https://doi.org/10.1007/s004660050314)]

  > An alternative elastodynamics BEM with no time history terms, but with more approximations and lower stability. Augmented with our new surface-only fictitious force terms similarly to CQBEM.

- Elastostatics BEM [[James and Pai 1999](https://doi.org/10.1145/311535.311542); [Hahn and Wojtan 2016](https://doi.org/10.1145/2897824.2925902)]

  > Elastostatics BEM with the null-space removal method [Hahn and Wojtan 2016].

- Rigid body

  > Simple rigid body simulator.

## Dependencies

The program is tested on a desktop machine running Ubuntu 20.04 with CUDA and an Apple Silicon Mac using gcc and clang compilers. Since Eigen is used with CUDA, [the compilation may fail with NVCC with MS Visual Studio](https://eigen.tuxfamily.org/dox/TopicCUDA.html).

### Required

- [Eigen]{https://eigen.tuxfamily.org/}
- [libigl]{https://libigl.github.io/}

  Eigen and libigl are included as git submodules. You can fetch them when you clone this repository by

        git clone --recurse-submodules https://github.com/rsugimoto/SurfaceOnlyDeformables

- OpenGL

### Optional

- OpenMP
- Intel MKL
- [OpenCCL](http://gamma.cs.unc.edu/COL/OpenCCL/download.html) and METIS

  Put the compiled OpenCCL binary and METIS binary under OpenCCL/Lib directory. You can get the complied binaries from the link above. When OpenCCL is not found, vertex reordering is disabled.

- CUDA

  CUDA is used to accelerate the precomputation process. When CUDA is not found, the program uses CPU for precomputation.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

You may want to change the CUDA architecture configuration in CMakeLists.txt before compilation.

## Run

We specify the options to the program using a json file. There are a few example scene configuration files in data directory, together with the required obj files. You can run the program as follows:

    build/main data/bunny_friction.json

## Citation

TBA

<!--   @article{Sugimoto2022:BEM,
        author = {Sugimoto, Ryusuke and Batty, Christopher and Hachisuka, Toshiya},
        title = {Surface-Only Dynamic Deformables using a Boundary Element Method},
        journal = {Computer Graphics Forum},
        volume = {41},
        number = {8},
        pages = {xx-xx},
        doi = {10.xxxxx/cgf.xxxxx},
        year = {2022}
    } -->
