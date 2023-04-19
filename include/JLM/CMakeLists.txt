
set(LLVM_TARGET_DEFINITIONS JLMDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRJLMDialectIncGen)
add_dependencies(mlir-headers MLIRJLMDialectIncGen)

set(LLVM_TARGET_DEFINITIONS JLMTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRJLMTypesIncGen)
add_dependencies(mlir-headers MLIRJLMTypesIncGen)

set(LLVM_TARGET_DEFINITIONS JLMOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRJLMOpsIncGen)
add_dependencies(mlir-headers MLIRJLMOpsIncGen)

add_mlir_doc(JLMOps JLMDialect Dialects/ -gen-dialect-doc)
