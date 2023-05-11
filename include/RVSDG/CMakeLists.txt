
set(LLVM_TARGET_DEFINITIONS RVSDGDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRRVSDGDialectIncGen)
add_dependencies(mlir-headers MLIRRVSDGDialectIncGen)

set(LLVM_TARGET_DEFINITIONS RVSDGInterfaces.td)
mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRRVSDGTypeInterfacesIncGen)
add_dependencies(mlir-headers MLIRRVSDGTypeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS RVSDGInterfaces.td)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRRVSDGOpInterfacesIncGen)
add_dependencies(mlir-headers MLIRRVSDGOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS RVSDGTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRRVSDGTypesIncGen)
add_dependencies(mlir-headers MLIRRVSDGTypesIncGen)

set(LLVM_TARGET_DEFINITIONS RVSDGAttrs.td)
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRRVSDGAttrsIncGen)
add_dependencies(mlir-headers MLIRRVSDGAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS RVSDGOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRRVSDGOpsIncGen)
add_dependencies(mlir-headers MLIRRVSDGOpsIncGen)

set(LLVM_TARGET_DEFINITIONS RVSDGPasses.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRRVSDGPassIncGen)
add_dependencies(mlir-headers MLIRRVSDGPassIncGen)

add_mlir_doc(RVSDGOps RVSDGDialect Dialects/ -gen-dialect-doc)
