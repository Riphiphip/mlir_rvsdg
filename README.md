# RVSDG MLIR dialect
Codebase for my master's thesis on implementing RVSDG as a dialect of MLIR

## Building
Clone this repository and its submodules.
```sh
git clone --recurse-submodules https://github.com/Riphiphip/mlir_rvsdg.git
``` 
If using a preinstalled version of LLVM and/or MLIR `--recurse-submodules` can be dropped.

### Building LLVM + MLIR (submodule)
Navigate to the `llvm` directory and run the following command:
```sh
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS='mlir' -DCMAKE_BUILD_TYPE=Debug
```

Without changing your working directory run
```sh
cmake --build ./build --target check-mlir
```

This will build the LLVM and MLIR libraries and associated tools.
