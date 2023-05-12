#ifndef RVSDG_INTERFACES
#define RVSDG_INTERFACES

include "RVSDG/RVSDGDialect.td"

def StateTypeInterface : TypeInterface<"StateTypeInterface"> {
    let description = [{
        Common interface for RVSDG state types.
    }];

    let cppNamespace = "::mlir::rvsdg"; 
}

def RVSDG_NodeInterface : OpInterface<"NodeInterface"> {
    let description = [{
        Common interface for RVSDG nodes. 
        All interface methods are named using RVSDG terminology.
    }];

    let cppNamespace = "::mlir::rvsdg";

    let methods = [
        InterfaceMethod< "Get node inputs", 
            "mlir::Operation::operand_range", "getRVSDGInputs", (ins), [{}],
            [{
                auto *op = this->getOperation();
                return op->getOperands();
            }]
        >,
        InterfaceMethod< "Get node outputs", 
            "mlir::Operation::result_range", "getRVSDGOutputs", (ins), [{}],
            [{
                auto *op = this->getOperation();
                return op->getResults();
            }]
        >,
    ];
}

def RVSDG_StructuralNodeInterface : OpInterface<"StructuralNodeInterface", [RVSDG_NodeInterface]> {
    let description = [{
        Common interface for RVSDG structural nodes. 
        All interface methods are named using RVSDG terminology.
    }];

    let cppNamespace = "::mlir::rvsdg";
}

#endif //RVSDG_INTERFACES