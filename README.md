# RVSDG MLIR dialect
MLIR dialect libraries for using the Regional Value State Dependence Graph (RVSDG) in MLIR. This repository contains the following:
- The RVSDG dialect library which is a representation of pure RVSDG in MLIR
    - TableGen files can be found under `include/RVSDG`
    - Header files can be found under `include/RVSDG`
    - C++ source files can be found under `lib/RVSDG`
- The JLM dialect library which builds on the RVSDG library and contains structures used for interoperation between RVSDG and JLM
    - TableGen files can be found under `include/JLM`
    - Header files can be found under `include/JLM`
    - C++ source files can be found under `lib/JLM`
- The `rvsdg-opt` tool which is a tool for parsing MLIR assembly that contains RVSDG and JLM dialects, and performing various transformations on the RVSDG
    - The tool can be found under `rvsdg-opt`
- The `rvsdg-lsp-server` tool which is an expansion of the MLIR language server that adds support for the RVSDG and JLM dialects
    - The tool can be found under `rvsdg-lsp-server`

## Software dependencies
 - LLVM 16.0.0
 - MLIR 16.0.0
 ### For building
 - Clang 16 
    - or another c++ compiler with support for `#pragma once` and C++17
 - CMake >= 3.13.4
 - Ninja-build
## Building the project
**NB!:** Build instructions have only been tested on ubuntu 22.04.

- Ensure all [software dependencies](#software-dependencies) have been installed and are working correctly
- Set the `LLVM_DIR` and `MLIR_DIR` environment variables to the paths containing the CMake directories for your LLVM and MLIR installations respectively.
    - The value for `LLVM_DIR` can often be found by running `llvm-config-16 --cmakedir`
    - The value for `MLIR_DIR` can often be found by running `llvm-config-16 --prefix` and appending `/lib/cmake/mlir` to the result
- Create a directory names `build` within the project root directory
- Run the following commands from inside the `build` directory:
```bash
cmake .. -GNinja
cmake --build .
```
- After the build has completed the library binaries can be found in `build/lib` and the `rvsdg-opt` and `rvsdg-lsp-server` tools can be found under `build/bin`

- Package containing library binaries and headers can be created by running 
```bash
ninja package
```


## Using dev containers
The `.devcontainer` directory contains a dockerfile that builds a docker container containing all required software for building and developing this project. The directory also contains the file `devcontainer.json` which can be used by the [Dev Container extension](https://code.visualstudio.com/docs/devcontainers/containers) for Visual Studio Code to set up a containerized development environment. 

Building the dev container will also build the `mlir_rvsdg` project and the `mlir-print` tool added to JLM. Both these can be found under `/home/docker` in the development container's file system.

Using this requires docker to be installed.
