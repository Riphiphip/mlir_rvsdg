#include "RVSDG/RVSDGPasses.h"
#include <mlir/Pass/PassManager.h>

namespace mlir::rvsdg::printExportPass
{
#define GEN_PASS_DEF_RVSDG_PRINTEXPORTPASS
#include "RVSDG/Passes.h.inc"
}

struct PrintExportPass : mlir::rvsdg::printExportPass::impl::RVSDG_PrintExportPassBase<PrintExportPass>
{

    PrintExportPass() = default;

    PrintExportPass(size_t nestingLevel) : nestingLevel(nestingLevel) {}

    PrintExportPass(const PrintExportPass &pass)
    {
        nestingLevel = pass.nestingLevel;
    }

    bool canScheduleOn(mlir::RegisteredOperationName opName) const override
    {
        return true;
    }

    void runOnOperation() override
    {
        if (!nestedPMInitialized)
        {
            nestedPM = initNestedPassManager();
            nestedPMInitialized = true;
        }
        mlir::Operation *op = getOperation();
        printf("%s", std::string(nestingLevel, '\t').c_str());
        printf("%s\n", op->getName().getStringRef().data());

        for (auto &region : op->getRegions())
        {
            for (auto &nested_op : region.getOps())
            {
                if (nested_op.hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
                {
                    if (failed(runPipeline(nestedPM, &nested_op)))
                    {
                        return signalPassFailure();
                    }
                }
                else
                {
                    printf("%s", std::string(nestingLevel + 1, '\t').c_str());
                    printf("%s\n", nested_op.getName().getStringRef().data());
                }
            }
        }
    }

private:
    size_t nestingLevel = 0;
    inline mlir::OpPassManager initNestedPassManager()
    {
        mlir::OpPassManager nestedPM;
        nestedPM.addPass(std::make_unique<PrintExportPass>(nestingLevel + 1));
        return nestedPM;
    }
    bool nestedPMInitialized = false;
    mlir::OpPassManager nestedPM;
};

std::unique_ptr<::mlir::Pass> createRVSDG_PrintExportPass()
{
    auto pass = std::make_unique<PrintExportPass>();
    return pass;
}