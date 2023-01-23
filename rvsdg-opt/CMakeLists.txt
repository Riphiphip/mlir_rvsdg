
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRRVSDG
        )
add_llvm_executable(rvsdg-opt rvsdg-opt.cpp)

llvm_update_compile_flags(rvsdg-opt)
target_link_libraries(rvsdg-opt PRIVATE ${LIBS})
