
set(LLVM_TARGET_DEFINITIONS RVSDGDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRRVSDGDialectIncGen)
add_dependencies(mlir-headers MLIRRVSDGDialectIncGen)

set(LLVM_TARGET_DEFINITIONS RVSDGTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRRVSDGTypesIncGen)
add_dependencies(mlir-headers MLIRRVSDGTypesIncGen)

set(LLVM_TARGET_DEFINITIONS RVSDGOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRRVSDGOpsIncGen)
add_dependencies(mlir-headers MLIRRVSDGOpsIncGen)

add_mlir_doc(RVSDGOps RVSDGDialect Dialects/ -gen-dialect-doc)
