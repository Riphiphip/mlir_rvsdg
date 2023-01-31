#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Types.h>
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/DialectImplementation.h"

#include "RVSDG/RVSDGDialect.h"
#include "RVSDG/RVSDGTypes.h"

#define GET_TYPEDEF_CLASSES
#include "RVSDG/Types.cpp.inc"

void mlir::rvsdg::RVSDGDialect::addRVSDGTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "RVSDG/Types.cpp.inc"
    >();
}