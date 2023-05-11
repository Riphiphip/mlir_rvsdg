
add_mlir_dialect_library(MLIRRVSDG
    RVSDGTypes.cpp
    RVSDGAttrs.cpp
    RVSDGDialect.cpp
    RVSDGOps.cpp
    RVSDGOpInterfaces.cpp
    
    RVSDGPrintExportPass.cpp

    DEPENDS
    MLIRRVSDGDialectIncGen
    MLIRRVSDGOpsIncGen
    MLIRRVSDGTypesIncGen
    MLIRRVSDGAttrsIncGen
    MLIRRVSDGOpInterfacesIncGen
    MLIRRVSDGPassIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    MLIRDialectUtils
)
