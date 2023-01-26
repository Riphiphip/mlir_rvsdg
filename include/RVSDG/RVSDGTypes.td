
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

def RVSDG_Type : AnyTypeOf<[StateType]>;

#endif //RVSDG_TYPES
