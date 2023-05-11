#pragma once
#include "mlir/IR/Region.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "RVSDG/RVSDGOpInterfaces.h"
#include "RVSDG/RVSDGTypes.h"
#include "RVSDG/RVSDGAttrs.h"

#define GET_OP_CLASSES
#include "RVSDG/Ops.h.inc"