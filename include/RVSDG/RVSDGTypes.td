
#ifndef RVSDG_TYPES
#define RVSDG_TYPES
include "mlir/IR/AttrTypeBase.td"

include "RVSDG/RVSDGDialect.td"
include "RVSDG/RVSDGInterfaces.td"


class RVSDG_Type<string name, string typeMnemonic, list<Trait> traits = []>: TypeDef<RVSDG_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

class RVSDG_StateType<string name, string typeMnemonic, list<Trait> traits = [StateTypeInterface]> : RVSDG_Type<name, typeMnemonic, traits>;

def RVSDG_IOState : RVSDG_StateType<"IOStateEdge", "ioState"> {
    let summary = "Type representing RVSDG state edges that sequentialize input/output operations";
    let description = [{
        SSA-values of this type are used to model RVSDG state edges
        that serialize input/output operations.
    }];
}

def RVSDG_MemState : RVSDG_StateType<"MemStateEdge", "memState"> {
    let summary = "Type representing RVSDG state edges that sequentialize memory operations";
    let description = [{
        SSA-values of this type are used to model RVSDG state edges
        that serialize memory operations.
    }];
}

def RVSDG_LoopState : RVSDG_StateType<"LoopStateEdge", "loopState"> {
    let summary = "Type representing RVSDG state edges that sequentialize loops TODO!: Figure out what this is for";
    let description = [{
        SSA-values of this type are used to model RVSDG state edges
        that serialize loops TODO!: Figure out what this is for.
    }];
}

def RVSDG_LambdaRef : RVSDG_Type<"LambdaRef", "lambdaRef"> {
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

def RVSDG_Ctrl : RVSDG_Type<"RVSDG_CTRL", "ctrl"> {
    let summary = "Type used as predicate for gamma and theta nodes";
    let description = [{
        Type used as predicate for gamma and theta nodes. Is generated
        by the match op. Has a fixed number of possible values.
    }];

    let parameters = (ins
        "uint64_t":$numOptions
    );

    let builders = [
        TypeBuilder<(ins "uint64_t":$numOptions)>
    ];

    let assemblyFormat = "`<` $numOptions `>`";
}

class RVSDG_Ctrl<int nOptions>: 
ConfinedType<RVSDG_Ctrl, 
    [CPred<"$_self.cast<::mlir::rvsdg::RVSDG_CTRLType>().getNumOptions() == " # nOptions # "">],
    "RVSDG control type with " # nOptions # " options"
>,
BuildableType<"::mlir::rvsdg::RVSDG_CTRLType::get($_builder.getContext(), " # nOptions # ")"> 
;

def RVSDG_Pointer : RVSDG_Type<"RVSDGPointer", "ptr"> {
    let summary = "Generic pointer type";
    let description = [{
        Generic pointer type. Mostly exists for interfacing with the JLM compiler.
        This type is also used for the output of delta nodes.
    }];

    let parameters = (ins
        OptionalParameter<"mlir::Type", "Type of the element pointed to by the pointer">:$elementType
    );

    let assemblyFormat = "(`<` $elementType^ `>`)?";
}

#endif //RVSDG_TYPES
