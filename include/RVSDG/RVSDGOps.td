include "mlir/IR/BuiltinTypes.td"

include "RVSDG/RVSDGDialect.td"
include "RVSDG/RVSDGTypes.td"

class RVSDG_Op<string mnemonic, list<Trait> traits = []> :
    Op<RVSDG_Dialect, mnemonic, traits>;

/**
*  Gamma Node
*/
def RVSDG_GammaNode: RVSDG_Op<"gammaNode", [IsolatedFromAbove]> {
    let summary = "RVSDG gamma-node which models conditional execution";
    let arguments = (ins
        Index:$predicate,
        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let regions = (region
        VariadicRegion<SizedRegion<1>>:$regions
    );

    let assemblyFormat = "attr-dict `(` $predicate `)` custom<TypedParamList>($inputs, type($inputs)) `:` custom<RVSDGRegions>($regions) `->` type(results)";
    let hasVerifier = 1;
}

def RVSDG_GammaResult: RVSDG_Op<"gammaResult", [
        HasParent<"mlir::rvsdg::GammaNode">,
        Terminator
    ]> {
    let summary = "Gamma node results";
    let description = [{
        Gamma node terminator. Maps operands to gamma node
        output.
    }];

    let arguments = (ins
        Variadic<AnyType>:$outputValues
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($outputValues, type($outputValues))";
    let hasVerifier = 1;
}


/**
*  Omega Node
*/
def RVSDG_OmegaNode: RVSDG_Op<"omegaNode", [IsolatedFromAbove]> {
    let summary = "RVSDG Omega node. Top level node in RVSDG";
    let arguments = (ins);
    let results = (outs);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "attr-dict custom<RVSDGRegion>($body)";
}

def RVSDG_OmegaResult: RVSDG_Op<"omegaResult", [
        HasParent<"mlir::rvsdg::OmegaNode">,
        Terminator
    ]> {
    let summary = "Omega node results";
    let description = [{
        Omega node terminator. Maps operands to items exported
        from the omega node.
    }];

    let arguments = (ins
        Variadic<AnyType>:$outputValues
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($outputValues, type($outputValues))";
}

/**
*  Lambda Node
*/

def RVSDG_LambdaNode: RVSDG_Op<"lambdaNode", [IsolatedFromAbove]> {

    let summary = "RVSDG lambda node. Models functions and methods.";

    let arguments = (ins
        Variadic<AnyType>: $contextValues
    );

    let results = (outs
        LambdaRef: $reference
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let assemblyFormat = "attr-dict type($reference) custom<TypedParamList>($contextValues, type($contextValues)) `:` custom<RVSDGRegion>($body)";

    let hasVerifier = 1;
}

def RVSDG_LambdaResult: RVSDG_Op<"lambdaResult", [
        HasParent<"mlir::rvsdg::LambdaNode">,
        Terminator
    ]> {
    let summary = "Lambda node results";
    let description = [{
        Lambda node terminator. Maps operands to outputs of apply
        node.
    }];

    let arguments = (ins
        Variadic<AnyType>:$outputValues
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($outputValues, type($outputValues))";

    let hasVerifier = 1;
}