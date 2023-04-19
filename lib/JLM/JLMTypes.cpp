#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Types.h>
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/DialectImplementation.h"

#include "JLM/JLMDialect.h"
#include "JLM/JLMTypes.h"

#define GET_TYPEDEF_CLASSES
#include "JLM/Types.cpp.inc"

void mlir::jlm::JLMDialect::addJLMTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "JLM/Types.cpp.inc"
    >();
}