
add_mlir_dialect_library(MLIRJLM
    JLMTypes.cpp
    JLMDialect.cpp
    JLMOps.cpp

    DEPENDS
    MLIRJLMDialectIncGen
    MLIRJLMOpsIncGen
    MLIRJLMTypesIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRRVSDG
    MLIRIR
    MLIRDialectUtils
)
