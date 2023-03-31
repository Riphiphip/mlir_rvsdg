#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Types.h>
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/DialectImplementation.h"

#include "RVSDG/RVSDGDialect.h"
#include "RVSDG/RVSDGAttrs.h"

#define GET_ATTRDEF_CLASSES
#include "RVSDG/Attrs.cpp.inc"

void mlir::rvsdg::RVSDGDialect::addRVSDGAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "RVSDG/Attrs.cpp.inc"
    >();
}