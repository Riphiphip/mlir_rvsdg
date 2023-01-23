
#include "RVSDG/RVSDGDialect.h"

#define GET_OP_CLASSES
#include "RVSDG/Ops.h.inc"


void mlir::rvsdg::RVSDGDialect::initialize(void){
    addOperations<
#define GET_OP_LIST
#include "RVSDG/Ops.cpp.inc"
    >();
}

#include "RVSDG/Dialect.cpp.inc"