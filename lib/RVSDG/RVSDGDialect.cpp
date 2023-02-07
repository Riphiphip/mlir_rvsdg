
#include "RVSDG/RVSDGDialect.h"

#include "RVSDG/RVSDGOps.h"
#include "RVSDG/RVSDGTypes.h"

void mlir::rvsdg::RVSDGDialect::initialize(void){
    addRVSDGTypes();
    addRVSDGOps();
}

#include "RVSDG/Dialect.cpp.inc"