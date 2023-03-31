
add_mlir_dialect_library(MLIRRVSDG
    RVSDGTypes.cpp
    RVSDGAttrs.cpp
    RVSDGDialect.cpp
    RVSDGOps.cpp

    DEPENDS
    MLIRRVSDGDialectIncGen
    MLIRRVSDGOpsIncGen
    MLIRRVSDGTypesIncGen
    MLIRRVSDGAttrsIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    MLIRDialectUtils
)
