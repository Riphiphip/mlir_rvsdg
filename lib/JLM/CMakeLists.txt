
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

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/JLM" DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/JLM" DESTINATION include
    FILES_MATCHING PATTERN "*.td"
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/build/include/JLM" DESTINATION include
    FILES_MATCHING PATTERN "*.h.inc"
)
