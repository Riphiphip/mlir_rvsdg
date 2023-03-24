
#ifndef RVSDG_TYPES
#define RVSDG_TYPES
include "mlir/IR/AttrTypeBase.td"

include "RVSDGDialect.td"
include "RVSDGInterfaces.td"


class RVSDG_Type<string name, string typeMnemonic, list<Trait> traits = []>: TypeDef<RVSDG_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

class RVSDG_StateType<string name, string typeMnemonic, list<Trait> traits = [StateTypeInterface]> : RVSDG_Type<name, typeMnemonic, traits>;

def IOStateType : RVSDG_StateType<"IOStateEdge", "ioState"> {
    let summary = "Type representing RVSDG state edges that sequentialize input/output operations";
    let description = [{
        SSA-values of this type are used to model RVSDG state edges
        that serialize input/output operations.
    }];
}

def MemStateType : RVSDG_StateType<"MemStateEdge", "memState"> {
    let summary = "Type representing RVSDG state edges that sequentialize memory operations";
    let description = [{
        SSA-values of this type are used to model RVSDG state edges
        that serialize memory operations.
    }];
}

def LoopStateType : RVSDG_StateType<"LoopStateEdge", "loopState"> {
    let summary = "Type representing RVSDG state edges that sequentialize loops TODO!: Figure out what this is for";
    let description = [{
        SSA-values of this type are used to model RVSDG state edges
        that serialize loops TODO!: Figure out what this is for.
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
        OptionalArrayRefParameter<"mlir::Type">:$parameterTypes,
        OptionalArrayRefParameter<"mlir::Type">:$returnTypes
    );
    
    let assemblyFormat = "`<` `(` ( `)` ):($parameterTypes^ `)` )? `->` `(` ( `)` ):($returnTypes^ `)` )? `>`";
}

def RVSDG_Pointer : RVSDG_Type<"RVSDGPointer", "rvsdgPointer"> {
    let summary = "Generic typed pointer type";
    let description = [{
        Generic typed pointer type. Mostly exists for interfacing with the JLM compiler.
        This type is also used for the output of delta nodes.
    }];

    let parameters = (ins
        TypeParameter<"mlir::Type", "Type of the element pointed to by the pointer">:$elementType
    );

    let assemblyFormat = "`<` $elementType `>`";
}

#endif //RVSDG_TYPES
