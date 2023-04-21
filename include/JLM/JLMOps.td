include "mlir/IR/BuiltinTypes.td"
include "RVSDG/RVSDGTypes.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

include "JLM/JLMDialect.td"
include "JLM/JLMTypes.td"


class JLM_Op<string mnemonic, list<Trait> traits = []> :
    Op<JLM_Dialect, mnemonic, traits>;


def JLM_Load: JLM_Op<"load", []> {
    let summary = "Load from memory";
    let description = [{
        Load from memory.
    }];

    let arguments = (ins
        LLVM_AnyPointer:$pointer,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        AnyType:$output,
        RVSDG_MemState:$outputMemState
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
        LLVM_AnyPointer:$pointer,
        AnyType:$value,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        RVSDG_MemState:$outputMemState
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
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        AnyTypeOf<[RVSDG_Pointer, LLVMPointerType]>:$output,
        RVSDG_MemState:$outputMemState
    );

    let assemblyFormat = "attr-dict $typeAttr `(` $inputMemStates `)` `->` type($output) `,` type($outputMemState)";
    let hasVerifier = 1;
}

def JLM_Memcpy: JLM_Op<"memcpy"> {
    let summary = "Copy memory content from one location to another";
    let description = [{
        Copy memory content from one location to another. Same parameters as LLVM memcpy,
        but also interacts takes and produces memory state values for use with RVSDG.
    }];

    let arguments = (ins 
        LLVM_AnyPointer:$dst,
        LLVM_AnyPointer:$src,
        AnySignlessInteger:$len,
        AnyTypeOf<[I1, RVSDG_Ctrl<2>]>: $isVolatile,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        RVSDG_MemState:$outputMemState
    );

    let assemblyFormat = "attr-dict $dst `:` type($dst) `,` $src `:` type($src) `,` $len `:` type($len) `,` $isVolatile `:` type($isVolatile) (`,` $inputMemStates^)? `->` type($outputMemState)";
}

def JLM_CreateVarArgList: JLM_Op<"createVarargs"> {
    let summary = "Create a list of variadic arguments";
    let description = [{
        Create a list of variadic arguments from severeal SSA values.
    }];

    let arguments = (ins
        Variadic<AnyType>:$input
    );

    let results = (outs
        JLM_VarargList:$output
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($input, type($input)) `->` type($output)";
}
