
#ifndef RVSDG_TYPES
#define RVSDG_TYPES
include "mlir/IR/AttrTypeBase.td"

include "RVSDGDialect.td"

class RVSDG_Type<string name, string typeMnemonic, list<Trait> traits = []>: TypeDef<RVSDG_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def StateType : RVSDG_Type<"StateEdge", "state"> {
    let summary = "Type representing RVSDG state edges";
    let description = [{
        SSA-values of this type are used to model RVSDG state edges.
        Any node that takes an input of this type should be 
        executed after the node that produced the value.
    }];
}

def LambdaRef : RVSDG_Type<"LambdaRef", "lambdaRef"> {
    let summary = "Reference to a lambda node";
    let description = [{
        SSA-values of this type represent a reference to a lambda
        node. They are produced as the outputs of lambda nodes. 
        A lambda node can be called by passing its produced lambdaRef
        to an apply node.
    }];

    let parameters = (ins
        ArrayRefParameter<"mlir::Type">:$argumentTypes,
        ArrayRefParameter<"mlir::Type">:$returnTypes
    );

    let assemblyFormat = "`<` $argumentTypes `->` $returnTypes `>`";
}

def DeltaRef : RVSDG_Type<"DeltaRef", "deltaRef"> {
    let summary = "Reference to a delta node";
    let description = [{
        SSA-values of this type represent a reference to a delta
        node. They are produced as the outputs of delta nodes. 
    }];
}

#endif //RVSDG_TYPES
