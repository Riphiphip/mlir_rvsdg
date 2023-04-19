
#ifndef RVSDG_ATTRS
#define RVSDG_ATTRS
include "mlir/IR/AttrTypeBase.td"

include "RVSDG/RVSDGDialect.td"


class RVSDG_Attr<string name, string typeMnemonic, list<Trait> traits = []>: AttrDef<RVSDG_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def RVSDG_MatchRuleAttr: RVSDG_Attr<"MatchRule", "matchRule"> {
    let summary = "Attribute for describing mappings for the match op";
    let description = [{
        This attribute is used to describe the mapping of the match op.
        Takes an integer value that is matched to another integer value.
    }];

    let parameters = (ins
        OptionalArrayRefParameter<"int64_t">: $values,
        "uint64_t": $index
    );

    let extraClassDeclaration = [{
        /**
        * Returns true if this mapping rule represents the default case.
        */
        bool isDefault() const { return this->getValues().empty(); }
    }];

    let assemblyFormat = "`<` ( `default` ):($values^)? `->` $index `>`";
}

#endif //RVSDG_ATTRS
