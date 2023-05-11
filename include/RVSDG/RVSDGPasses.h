#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

#include "RVSDG/RVSDGDialect.h"
#include "RVSDG/RVSDGOps.h"

#define GEN_PASS_DECL
#include "RVSDG/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "RVSDG/Passes.h.inc"