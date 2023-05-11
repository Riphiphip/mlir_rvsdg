#ifndef RVSDG_INTERFACES
#define RVSDG_INTERFACES

include "RVSDG/RVSDGDialect.td"

def StateTypeInterface : TypeInterface<"StateTypeInterface"> {
    let description = [{
        Common interface for RVSDG state types.
    }];

    let cppNamespace = "::mlir::rvsdg"; 
}

def RVSDG_StructuralNodeInterface : OpInterface<"StructuralNodeInterface"> {
    let description = [{
        Common interface for RVSDG structural nodes. 
        All interface methods are named using RVSDG terminology.
    }];

    let cppNamespace = "::mlir::rvsdg";

    let methods = [
        InterfaceMethod< "Get node inputs", 
            "mlir::Operation::operand_range", "getRVSDGInputs", (ins), [{}],
            [{return this->getOperands();}]
        >,
        InterfaceMethod< "Get node output", 
            "mlir::Operation::result_range", "getRVSDGOutputs", (ins), [{}],
            [{return this->getResults();}]
        >,
    ];
}

#endif //RVSDG_INTERFACES