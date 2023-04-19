
#ifndef JLM_TYPES
#define JLM_TYPES
include "mlir/IR/AttrTypeBase.td"

include "JLMDialect.td"


class JLM_Type<string name, string typeMnemonic, list<Trait> traits = []>: TypeDef<JLM_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}


#endif //JLM_TYPES
