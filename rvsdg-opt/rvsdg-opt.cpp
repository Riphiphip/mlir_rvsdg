#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include <stdio.h>

#include <RVSDG/RVSDGDialect.h>

int main(int argc, char *argv[]) {
  printf("Hello from rvsdg-opt!\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::rvsdg::RVSDGDialect>();
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "RVSDG Optimizer driver", registry)
  );
}