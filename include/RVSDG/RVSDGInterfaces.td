#ifndef RVSDG_INTERFACES
#define RVSDG_INTERFACES

include "RVSDG/RVSDGDialect.td"

def StateTypeInterface : TypeInterface<"StateTypeInterface"> {
    let description = [{
        Common interface for RVSDG state types.
    }];

    let cppNamespace = "::mlir::rvsdg"; 
}

#endif //RVSDG_INTERFACES