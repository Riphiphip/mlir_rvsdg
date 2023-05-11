FROM ubuntu:22.04

USER root

RUN useradd -ms /bin/bash docker
RUN usermod -a -G sudo docker
    
RUN apt update -y -qq &&\
    apt upgrade -y -qq &&\
    apt install -y -qq \
    	sudo \
        build-essential \
        cmake \
        git \
        ninja-build \
        gnupg \
        wget

RUN yes | unminimize

#Install LLVM 14 and 16. Both are required for the build.
RUN printf "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-14 main\n" >> /etc/apt/sources.list &&\
    printf "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-14 main\n" >> /etc/apt/sources.list &&\
    printf "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main\n" >> /etc/apt/sources.list &&\
    printf "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main\n" >> /etc/apt/sources.list
RUN (wget -O - "https://apt.llvm.org/llvm-snapshot.gpg.key" | apt-key add -)
RUN apt update -y -qq
RUN apt-get install -qq -y libllvm-14-ocaml-dev libllvm14 llvm-14 llvm-14-dev llvm-14-doc llvm-14-examples llvm-14-runtime
RUN apt-get install -qq -y libllvm-16-ocaml-dev libllvm16 llvm-16 llvm-16-dev llvm-16-doc llvm-16-examples llvm-16-runtime
RUN apt-get install -qq -y clang-14 clang-tools-14 clang-14-doc libclang-common-14-dev libclang-14-dev libclang1-14 clang-format-14 python3-clang-14 clangd-14 clang-tidy-14
RUN apt-get install -qq -y clang-16 clang-tools-16 clang-16-doc libclang-common-16-dev libclang-16-dev libclang1-16 clang-format-16 python3-clang-16 clangd-16 clang-tidy-16 lldb-16
# Only MLIR 16 is required
RUN apt-get install -qq -y libmlir-16-dev mlir-16-tools

USER docker

# Download and build dialect libraries
WORKDIR /home/docker
RUN git clone https://github.com/Riphiphip/mlir_rvsdg.git
RUN mkdir build
WORKDIR /home/docker/mlir_rvsdg/build
ENV CXX clang++-16
ENV LLVM_DIR /usr/lib/llvm-16/lib/cmake/llvm
ENV MLIR_DIR /usr/lib/llvm-16/lib/cmake/mlir
RUN cmake .. -GNinja
RUN cmake --build .

# Download and build mlir printer
WORKDIR /home/docker
RUN git clone https://github.com/Riphiphip/jlm.git
WORKDIR /home/docker/jlm
RUN git checkout mlir-print
RUN make mlir-print-debug

WORKDIR /home/docker
ENTRYPOINT [ "/bin/bash" ]
