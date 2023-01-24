
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRLspServerLib
        MLIRRVSDG
        )
add_llvm_executable(rvsdg-lsp-server rvsdg-lsp-server.cpp)

llvm_update_compile_flags(rvsdg-lsp-server)
target_link_libraries(rvsdg-lsp-server PRIVATE ${LIBS})
