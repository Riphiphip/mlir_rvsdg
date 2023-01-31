
add_mlir_dialect_library(MLIRRVSDG
    RVSDGTypes.cpp
    RVSDGDialect.cpp
    RVSDGOps.cpp

    DEPENDS
    MLIRRVSDGDialectIncGen
    MLIRRVSDGOpsIncGen
    MLIRRVSDGTypesIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    MLIRDialectUtils
)
