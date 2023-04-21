#ifndef JLM_BASE
#define JLM_BASE
include "mlir/IR/OpBase.td"

def JLM_Dialect : Dialect {
  let name = "jlm";
  let cppNamespace = "::mlir::jlm";
  let description = [{
    Dialect for interoperation with the JLM compiler.
    Contains JLM specific types and operations.
  }];
  let useFoldAPI = kEmitFoldAdaptorFolder;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
  /*
  * Used for registering JLM specific parametrized types.
  * Must be defined as a part of the same unit as type class definitions.
  */
  void addJLMTypes();

  /*
  * Registers JLM specific Ops.
  */
  void addJLMOps();
  }];
}
#endif // JLM_BASE