
#include "RVSDG/RVSDGDialect.h"

#include "RVSDG/RVSDGOps.h"
#include "RVSDG/RVSDGTypes.h"

void mlir::rvsdg::RVSDGDialect::initialize(void){
    addRVSDGTypes();
    addOperations<
#define GET_OP_LIST
#include "RVSDG/Ops.cpp.inc"
    >();
}

#include "RVSDG/Dialect.cpp.inc"