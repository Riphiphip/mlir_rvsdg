#ifndef RVSDG_BASE
#define RVSDG_BASE
include "mlir/IR/OpBase.td"

def RVSDG_Dialect : Dialect {
  let name = "rvsdg";
  let cppNamespace = "::mlir::rvsdg";
  let description = [{
    Regionalized Value State Dependency Graph
  }];
  let useFoldAPI = kEmitFoldAdaptorFolder;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
  /*
  * Used for registering custom parametrized types.
  * Must be defined as a part of the same unit as class definitions.
  */
  void addRVSDGTypes();
  }];
}
#endif // RVSDG_BASE