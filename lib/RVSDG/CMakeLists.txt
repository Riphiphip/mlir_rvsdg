
add_mlir_dialect_library(MLIRRVSDG
    RVSDGDialect.cpp
    RVSDGOps.cpp
    RVSDGTypes.cpp

    DEPENDS
    MLIRRVSDGDialectIncGen
    MLIRRVSDGOpsIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    MLIRDialectUtils
)
