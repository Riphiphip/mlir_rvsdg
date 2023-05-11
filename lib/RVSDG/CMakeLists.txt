
add_mlir_dialect_library(MLIRRVSDG
    RVSDGTypes.cpp
    RVSDGAttrs.cpp
    RVSDGDialect.cpp
    RVSDGOps.cpp
    
    RVSDGPrintExportPass.cpp

    DEPENDS
    MLIRRVSDGDialectIncGen
    MLIRRVSDGOpsIncGen
    MLIRRVSDGTypesIncGen
    MLIRRVSDGAttrsIncGen
    MLIRRVSDGPassIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    MLIRDialectUtils
)
