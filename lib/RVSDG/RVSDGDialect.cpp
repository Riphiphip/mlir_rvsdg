
#include "RVSDG/RVSDGDialect.h"

#include "RVSDG/RVSDGOps.h"
#include "RVSDG/RVSDGTypes.h"

void mlir::rvsdg::RVSDGDialect::initialize(void){
    addTypes<
#define GET_TYPEDEF_LIST
#include "RVSDG/Types.cpp.inc"
    >();
    addOperations<
#define GET_OP_LIST
#include "RVSDG/Ops.cpp.inc"
    >();
}

#include "RVSDG/Dialect.cpp.inc"