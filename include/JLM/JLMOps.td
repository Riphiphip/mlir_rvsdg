include "mlir/IR/BuiltinTypes.td"
include "RVSDG/RVSDGTypes.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

include "JLM/JLMDialect.td"


class JLM_Op<string mnemonic, list<Trait> traits = []> :
    Op<JLM_Dialect, mnemonic, traits>;


def JLM_Load: JLM_Op<"load"> {
    let summary = "Load from memory";
    let description = [{
        Load from memory.
    }];

    let arguments = (ins
        AnyTypeOf<[RVSDG_Pointer, LLVMPointerType]>:$pointer,
        Variadic<MemStateType>:$inputMemStates
    );

    let results = (outs
        AnyType:$output,
        MemStateType:$outputMemState
    );

    let assemblyFormat = "attr-dict $pointer `:` type($pointer) `(` $inputMemStates `)` `->` type($output) `,` type($outputMemState)";
    let hasVerifier = 1;
}

def JLM_Store: JLM_Op<"store"> {
    let summary = "Store to memory";
    let description = [{
        Store to memory.
    }];

    let arguments = (ins
        AnyTypeOf<[RVSDG_Pointer, LLVMPointerType]>:$pointer,
        AnyType:$value,
        Variadic<MemStateType>:$inputMemStates
    );

    let results = (outs
        MemStateType:$outputMemState
    );

    let assemblyFormat = "attr-dict `(` $pointer `:` type($pointer) `,` $value `:` type($value) `)` `(` $inputMemStates `)` `->` type($outputMemState)";
    let hasVerifier = 1;
}

def JLM_Alloca: JLM_Op<"alloca"> {
    let summary = "Allocate memory on the stack";
    let description = [{
        Allocate memory on the stack.
    }];

    let arguments = (ins
        TypeAttr:$typeAttr,
        Variadic<MemStateType>:$inputMemStates
    );

    let results = (outs
        AnyTypeOf<[RVSDG_Pointer, LLVMPointerType]>:$output,
        MemStateType:$outputMemState
    );

    let assemblyFormat = "attr-dict $typeAttr `(` $inputMemStates `)` `->` type($output) `,` type($outputMemState)";
    let hasVerifier = 1;
}
