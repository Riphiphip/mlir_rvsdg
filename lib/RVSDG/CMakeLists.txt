
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

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/RVSDG" DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/RVSDG" DESTINATION include
    FILES_MATCHING PATTERN "*.td"
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/build/include/RVSDG" DESTINATION include
    FILES_MATCHING PATTERN "*.h.inc"
)
