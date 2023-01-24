
#include "RVSDG/RVSDGDialect.h"

#include "RVSDG/RVSDGOps.h"


void mlir::rvsdg::RVSDGDialect::initialize(void){
    addOperations<
#define GET_OP_LIST
#include "RVSDG/Ops.cpp.inc"
    >();
}

#include "RVSDG/Dialect.cpp.inc"