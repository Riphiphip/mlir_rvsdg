
#ifndef JLM_TYPES
#define JLM_TYPES
include "mlir/IR/AttrTypeBase.td"

include "JLMDialect.td"


class JLM_Type<string name, string typeMnemonic, list<Trait> traits = []>: TypeDef<JLM_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def JLM_VarargList: JLM_Type<"VarargList", "varargList"> {
    let summary = "JLM VarargList type";
    let description = [{
        The JLM VarargList type represents a list of arguments that are
        passed as varargs to a function. Since this type models C varargs,
        it is not possible to know the number or types of arguments at 
        compile time.
    }];
}

#endif //JLM_TYPES
