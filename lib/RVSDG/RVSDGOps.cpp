#include "mlir/IR/Block.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "RVSDG/RVSDGDialect.h"
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
  if (parent == NULL) {
    return emitOpError(
        "GammaResult has no parent of type GammaNode. This error should never "
        "appear, so if it does, may God have mercy on your soul");
  }

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
 * Lambda node
 */

/**
 * @brief Verifies structure of lambda node.
 * Verifies the following attributes:
 *  - Number and types of region arguments when compared to inputs
 *    - Given n inputs with types t1, t2 ... tn, there should be at
 *      least n region arguments where the first n arguments should
 *      have types t1, t2 ... tn.
 *  - Verify that function signature has the same number of inputs
 *    as the lambda node.
 *
 */
LogicalResult LambdaNode::verify() {
  auto bodyArgumentTypes = this->getRegion().getArgumentTypes();
  auto nodeInputTypes = this->getOperandTypes();

  if (bodyArgumentTypes.size() < nodeInputTypes.size()) {
    return emitOpError(
               "Number of arguments to lambda body needs to be greater than or "
               "equal to number of node inputs. Expected at least ")
           << nodeInputTypes.size() << " argument(s), got "
           << bodyArgumentTypes.size();
  }
  for (size_t i = 0; i < nodeInputTypes.size(); ++i) {
    if (nodeInputTypes[i] != bodyArgumentTypes[i]) {
      return emitOpError("Mismatched types in lambda node body arguments. "
                         "First arguments "
                         "should match node inputs. "
                         "Offending argument: #")
             << i << " Expected " << nodeInputTypes[i] << ", got "
             << bodyArgumentTypes[i];
    }
  }

  Value signatureVal = this->getResult();
  if (!signatureVal.getType().isa<LambdaRefType>()) {
    return emitOpError(
        "Result type is invalid. Expected an instance of LambdaRefType. If you "
        "see this error I feel bad for you.");
  }
  LambdaRefType signatureType = signatureVal.getType().cast<LambdaRefType>();
  ArrayRef<Type> signatureParamTypes = signatureType.getParameterTypes();

  if (signatureParamTypes.size() !=
      bodyArgumentTypes.size() - nodeInputTypes.size()) {
    return emitOpError("Mismatch between lambda signature and body arguments: "
                       "Number of arguments in signature: ")
           << signatureParamTypes.size()
           << ". Number of arguments in body AFTER context values: "
           << bodyArgumentTypes.size() - nodeInputTypes.size();
  }

  for (size_t sigI = 0, argI = nodeInputTypes.size();
       sigI < signatureParamTypes.size(); ++sigI, ++argI) {
    if (signatureParamTypes[sigI] != bodyArgumentTypes[argI]) {
      return emitOpError("Mismatched types between lambda signature and body "
                         "arguments. ")
             << "Signature argument #" << sigI << " has type "
             << signatureParamTypes[sigI] << ". Body argument #" << argI
             << " has type " << bodyArgumentTypes[argI];
    }
  }
  return LogicalResult::success();
}

LogicalResult LambdaResult::verify() {
  auto parent = dyn_cast<LambdaNode>((*this)->getParentOp());
  if (parent == NULL) {
    return emitOpError(
        "LambdaResult has no parent of type LambdaNode. This error should "
        "never appear, so if it does, may God have mercy on your soul");
  }
  Value signatureVal = parent.getResult();
  if (!signatureVal.getType().isa<LambdaRefType>()) {
    return emitOpError(
        "Result type is invalid. Expected an instance of LambdaRefType. If you "
        "see this error I feel bad for you.");
  }
  LambdaRefType signatureType = signatureVal.getType().cast<LambdaRefType>();
  ArrayRef<Type> signatureReturnTypes = signatureType.getReturnTypes();

  auto resultTypes = this->getOperandTypes();

  if (signatureReturnTypes.size() != resultTypes.size()) {
    return emitOpError("Number of operands to lambda terminator does not match "
                       "number of return types in signature.");
  }

  size_t typeIndex = 0;
  for (auto [sigType, resType] : zip(signatureReturnTypes, resultTypes)) {
    if (sigType != resType) {
      return emitOpError("Type mismatch between lambda signature and lambda "
                         "result Op. Offending type: #")
             << typeIndex << " Signature has type " << sigType
             << ", result Op has type " << resType;
    }
    ++typeIndex;
  }

  return LogicalResult::success();
}

/**
 * Theta node
 */

LogicalResult ThetaNode::verify() {
  auto inputTypes = this->getInputs().getTypes();
  auto outputTypes = this->getOutputs().getTypes();
  auto regionArgTypes = this->getRegion().getArgumentTypes();

  if (inputTypes.size() != outputTypes.size() ||
      inputTypes.size() != regionArgTypes.size()) {
    return emitOpError(" has should have an equal number of inputs, outputs,"
                       " and region arguments.")
           << " Number of inputs: " << inputTypes.size()
           << " Number of outputs: " << outputTypes.size()
           << " Number of region arguments: " << regionArgTypes.size();
  }
  size_t typeIndex = 0;
  for (auto [inType, outType, argType] :
       zip(inputTypes, outputTypes, regionArgTypes)) {
    if (inType != outType || inType != argType) {
      return emitOpError("Type mismatch between node inputs, node outputs, and "
                         "region arguments. "
                         "Offending argument: #")
             << typeIndex << " Input type: " << inType
             << " Output type: " << outType
             << " Region argument type: " << argType;
    }
    ++typeIndex;
  }
  return LogicalResult::success();
}

LogicalResult ThetaResult::verify() {
  auto resultTypes = this->getOutputValues().getTypes();
  ThetaNode parent = dyn_cast<ThetaNode>((*this)->getParentOp());
  if (parent == NULL) {
    return emitOpError(
        "ThetaResult has no parent of type ThetaNode. This error should never "
        "appear, so if it does, may God have mercy on your soul");
  }
  auto thetaOperandTypes = parent.getOperandTypes();

  if (parent.getNumOperands() != this->getOutputValues().size()) {
    return emitOpError(" should have a number of non-predicate operands equal "
                       "to the number of inputs in the parent theta node.")
           << " Number of operands: " << resultTypes.size()
           << " Number of inputs: " << parent.getNumOperands();
  }

  size_t typeIndex = 0;
  for (auto [inType, resType] : zip(thetaOperandTypes, resultTypes)) {
    if (inType != resType) {
      return emitOpError("Type mismatch between theta inputs and theta "
                         "result Op. Offending type: #")
             << typeIndex << " Input has type " << inType
             << ", result Op has type " << resType;
    }
    ++typeIndex;
  }

  return LogicalResult::success();
}


/**
 * Phi node
 */

LogicalResult PhiNode::verify() {
  size_t nOperands = this->getNumOperands();
  size_t nOutputs = this->getOutputs().size();
  size_t nArgs = this->getRegion().getNumArguments();
  if (nOperands + nOutputs != nArgs) {
    return this->emitOpError(" has wrong number of region arguments. Number of arguments should be equal to the number of inputs plus the number of outputs.");
  }

  for (auto [operandType, argType]: zip(this->getOperandTypes(), this->getRegion().getArgumentTypes())) {
    if (operandType != argType) {
      this->emitOpError(" has a type mismatch between inputs and region arguments");
    }
  }

  auto argTypes = this->getRegion().getArgumentTypes();
  auto outputs = this->getOutputs();

  for (size_t outputIndex=0, argIndex=nOperands; outputIndex < nOutputs; ++outputIndex, ++argIndex) {
    if (outputs[outputIndex].getType() != argTypes[argIndex]) {
      return this->emitOpError(" has a type mismatch between outputs and region arguments");
    }
  }

  return LogicalResult::success();
}

LogicalResult PhiResult::verify() {
  PhiNode parent = dyn_cast<PhiNode>((*this)->getParentOp());
  if (parent == NULL) {
    return emitOpError(
        "PhiResult has no parent of type PhiNode. This error should never "
        "appear, so if it does, may God have mercy on your soul");
  }
  auto resultTypes = this->getOperandTypes();
  const auto &outputs = parent.getOutputs();

  for (size_t i=0; i < resultTypes.size(); ++i) {
    if (resultTypes[i] != outputs[i].getType()) {
      return this->emitOpError(" has a type mismatch between result Op and node outputs");
    }
  }
  /* 
  Should technically check type matchup with region arguments as well, but
  match between outputs and region arguments is already being checked in the
  PhiNode verifier.
  */
  return LogicalResult::success();
}

/**
 * Delta node
 */

LogicalResult DeltaNode::verify() {
  auto inputTypes = this->getInputs().getTypes();
  auto regionArgTypes = this->getRegion().getArgumentTypes();

  if (inputTypes.size() != regionArgTypes.size()) {
    return emitOpError(" should have an equal number of inputs and region arguments.")
           << " Number of inputs: " << inputTypes.size()
           << " Number of region arguments: " << regionArgTypes.size();
  }
  size_t typeIndex = 0;
  for (auto [inType, argType] :zip(inputTypes, regionArgTypes)) {
    if (inType != argType) {
      return emitOpError("Type mismatch between node inputs and region arguments.")
            << " Offending argument: #"
            << typeIndex 
            << " Input type: " << inType
            << " Region argument type: " << argType;
    }
    ++typeIndex;
  }
  return LogicalResult::success();
}

LogicalResult DeltaResult::verify() {
  auto parent = dyn_cast<DeltaNode>((*this)->getParentOp());
  if (parent == NULL) {
    return emitOpError("DeltaResult has no parent of type DeltaNode.");
  }
  auto resultType = this->getOperand().getType();
  auto outputElementType = parent.getOutput().getType().getElementType();
  if (resultType != outputElementType) {
    return emitOpError("Type mismatch between DeltaResult and DeltaNode output.")
           << " DeltaResult type: " << resultType
           << " DeltaNode output element type: " << outputElementType;
  }
  return LogicalResult::success();
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
    regions.push_back(std::move(region));
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

/**
 * Implement dialect method for registering Ops
 */
void mlir::rvsdg::RVSDGDialect::addRVSDGOps() {
  addOperations<
#define GET_OP_LIST
#include "RVSDG/Ops.cpp.inc"
      >();
}
