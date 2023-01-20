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
}
#endif // RVSDG_BASE