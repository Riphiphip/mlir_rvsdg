
#include "JLM/JLMDialect.h"

#include "JLM/JLMOps.h"
#include "JLM/JLMTypes.h"

void mlir::jlm::JLMDialect::initialize(void){
    addJLMTypes();
    addJLMOps();
}

#include "JLM/Dialect.cpp.inc"