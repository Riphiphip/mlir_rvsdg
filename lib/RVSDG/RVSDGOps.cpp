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
 *  - Number and type of region arguments (should match gamma inputs)
 */
LogicalResult GammaNode::verify() {
  if (this->getNumRegions() < 2) {
    return emitOpError("has too few regions. Minimum number of regions is 2, "
                       "but Op has ")
           << this->getNumRegions();
  }
  for (auto &region : this->getRegions()) {
    if (region.getNumArguments() != this->getInputs().size()) {
      return emitOpError(" has region with wrong number of arguments. "
                         "Offending region: #")
             << region.getRegionNumber() << ". Expected "
             << this->getInputs().size() << ", got "
             << region.getNumArguments();
    }
    auto arguments = region.getArguments();
    auto inputs = this->getInputs();
    for (size_t i = 0; i < region.getNumArguments(); ++i) {
      if (arguments[i].getType() != inputs[i].getType()) {
        auto argument = arguments[i];
        emitOpError(" has mismatched region argument types: Region #")
            << region.getRegionNumber() << " Argument #"
            << argument.getArgNumber() << ". Expected " << inputs[i].getType()
            << ", got " << arguments[i].getType();
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

/**
 * @brief Prints a list of regions prefixed with a list of region arguments
 * and their types
 *
 * @param p Assembly printer
 * @param op Operation which we are printing
 * @param regions Regions of the operation
 **/
void printRVSDGRegion(OpAsmPrinter &p, Operation *op, Region &region) {
  p << "(";
  size_t argument_count = region.getNumArguments();
  for (size_t argument_index = 0; argument_index < argument_count;
       argument_index++) {
    if (argument_index != 0) {
      p << ", ";
    }
    p.printRegionArgument(region.getArgument(argument_index));
  }
  p << "): ";
  p.printRegion(region, false, true, true);
}

void printRVSDGRegions(OpAsmPrinter &p, Operation *op,
                       MutableArrayRef<Region> regions) {
  p.increaseIndent();
  p << "[";
  p.printNewline();
  size_t region_count = regions.size();
  for (size_t region_index = 0; region_index < region_count; ++region_index) {
    if (region_index != 0) {
      p << ", ";
      p.printNewline();
    }
    printRVSDGRegion(p, op, regions[region_index]);
  }
  p.decreaseIndent();
  p.printNewline();
  p << "]";
}

ParseResult parseRVSDGRegion(OpAsmParser &parser, Region &region) {
  SmallVector<OpAsmParser::Argument, 4> arguments;
  if (failed(parser.parseArgumentList(arguments, OpAsmParser::Delimiter::Paren,
                                      true, true))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "Failed to parse argument list");
  }
  if (failed(parser.parseColon())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected a \":\" token");
  }

  if (failed(parser.parseRegion(region, arguments, true))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "Failed to parse region");
  }
  return ParseResult::success();
}

ParseResult
parseRVSDGRegions(OpAsmParser &parser,
                  SmallVectorImpl<std::unique_ptr<Region>> &regions) {
  auto parseRegion = [&]() -> ParseResult {
    std::unique_ptr<Region> region = std::make_unique<Region>();
    if (failed(parseRVSDGRegion(parser, *region))) {
      return ParseResult::failure();
    }
    regions.push_back(move(region));
    return ParseResult::success();
  };

  ParseResult result = parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Square, parseRegion);
  if (failed(result)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "Failed to parse region list");
  }
  return ParseResult::success();
}

/**
 * Auto generated sources
 */
#define GET_OP_CLASSES
#include "RVSDG/Ops.cpp.inc"