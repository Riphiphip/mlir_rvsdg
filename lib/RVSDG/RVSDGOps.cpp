#include "mlir/IR/Block.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "RVSDG/RVSDGOps.h"

using namespace mlir;
using namespace rvsdg;

/**
 * Gamma node implementations
 */

/**
 * @brief Verifies structure of built gamma node.
 * Verifies the following attributes:
 *  - Number of regions (>= 2)
 *  - Number of blocks in each region (1)
 *  - Number and type of block arguments (should match gamma inputs)
 */
LogicalResult GammaNode::verify() {
  if (this->getNumRegions() < 2) {
    return emitOpError("has too few regions. Minimum number of regions is 2, "
                       "but Op has ")
           << this->getNumRegions();
  }
  for (auto region : this->getRegions()) {
    size_t blockCount = region->getBlocks().size();
    if (blockCount != 1) {
      return emitOpError(
                 " has wrong number of blocks in region. Expected 1, got ")
             << blockCount;
    }
    for (auto &block : region->getBlocks()) {
      if (block.getNumArguments() != this->getInputs().size()) {
        return emitOpError(
                   " has block with wrong number of arguments in region #")
               << region->getRegionNumber() << ". Expected "
               << this->getInputs().size() << ", got "
               << block.getNumArguments();
      }
      auto arguments = block.getArguments();
      auto inputs = this->getInputs();
      for (size_t i = 0; i < block.getNumArguments(); ++i) {
        if (arguments[i].getType() != inputs[i].getType()) {
          emitOpError(" has mismatched block argument types. Expected")
              << inputs[i].getType() << ", got " << arguments[i].getType();
        }
      }
    }
  }
  return LogicalResult::success();
}

/**
 * Gamma node terminator implementations
 */
LogicalResult GammaResult::verify() {
  auto parent = cast<GammaNode>((*this)->getParentOp());
  const auto &results = parent.getResults();
  if (getNumOperands() != results.size()) {
    return emitOpError("has ")
           << getNumOperands() << " operands, but parent node outputs "
           << results.size();
  }

  for (size_t i = 0; i < results.size(); ++i) {
    if (getOperand(i).getType() != results[i].getType()) {
      return emitError() << "type of output operand " << i << " ("
                         << getOperand(i).getType()
                         << ") does not match node output type ("
                         << results[i].getType() << ")";
    }
  }

  return success();
}

/**
 * Assembly directives
 */

/**
 * @brief Prints out a comma separated list of parameters paired with their
 * respective types. Having types as a parameter is redundant, but tablegen
 * won't build without it.
 *
 * @param p Assembly printer
 * @param op Operation which we are printing
 * @param operands Range of operands to be printed
 * @param types Types of the operands will be matched with operands using
 *              position in the array
 **/
void printTypedParamList(OpAsmPrinter &p, Operation *op, OperandRange operands,
                         TypeRange types) {
  p << "(";
  int param_count = std::min(operands.size(), types.size());
  for (int i = 0; i < param_count; ++i) {
    if (i != 0) {
      p << ", ";
    }
    p.printOperand(operands[i]);
    p << ": ";
    p.printType(types[i]);
  }
  p << ")";
}

ParseResult
parseTypedParamList(OpAsmParser &parser,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                    SmallVectorImpl<Type> &types) {

  if (parser.parseLParen().failed()) {
    return ParseResult::failure();
  }
  unsigned int index = 0;
  while (parser.parseOptionalRParen().failed()) {
    if (index != 0) {
      if (parser.parseComma().failed()) {
        return ParseResult::failure();
      }
    }
    mlir::OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand).failed()) {
      return ParseResult::failure();
    }
    Type type;
    if (parser.parseColonType(type).failed()) {
      return ParseResult::failure();
    }
    operands.push_back(operand);
    types.push_back(type);
    ++index;
  }

  return ParseResult::success();
}

#define GET_OP_CLASSES
#include "RVSDG/Ops.cpp.inc"