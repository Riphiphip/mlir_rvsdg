#include <unordered_set>

#include "mlir/IR/Block.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "RVSDG/RVSDGDialect.h"
#include "RVSDG/RVSDGOps.h"
#include "RVSDG/RVSDGASMDirectives.h"

using namespace mlir;
using namespace rvsdg;

/**
 * Gamma node implementations
 */

/**
 * @brief Verifies structure of built gamma node.
 * Verifies the following attributes:
 *  - Number of regions (>= 2)
 *  - Number of options in predicate (should match number of regions)
 *  - Number and type of region arguments (should match gamma inputs)
 */
LogicalResult GammaNode::verify() {
  if (this->getNumRegions() < 2) {
    return emitOpError("has too few regions. Minimum number of regions is 2, "
                       "but Op has ")
           << this->getNumRegions();
  }
  auto predicateType = this->getPredicate().getType();
  if (predicateType.getNumOptions() != this->getNumRegions()){
    return emitOpError("has predicate with wrong number of options. Expected ")
           << this->getNumRegions() << ", got "
           << predicateType.getNumOptions();
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
 *  - Number and types of region arguments when compared to inputs and function signature
 *    - Given a function signature with n operands of types T1, T2, ..., Tn, and k inputs 
 *      of types I1, I2, ..., Ik, the lambda node region should have n+k arguments of 
 *      types T1, T2, ..., Tn, I1, I2, ..., Ik.
 *
 */
LogicalResult LambdaNode::verify() {

  Value signatureVal = this->getResult();
  if (!signatureVal.getType().isa<LambdaRefType>()) {
    return emitOpError(
        "Result type is invalid. Expected an instance of LambdaRefType.");
  }
  LambdaRefType signatureType = signatureVal.getType().cast<LambdaRefType>();

  ArrayRef<Type> signatureParamTypes = signatureType.getParameterTypes();
  auto bodyArgumentTypes = this->getRegion().getArgumentTypes();
  auto nodeInputTypes = this->getOperandTypes();

  if (bodyArgumentTypes.size() != signatureParamTypes.size() + nodeInputTypes.size()) {
    return emitOpError("Mismatched number of arguments between lambda signature and body. ")
           << "Signature has " << signatureParamTypes.size() << " arguments\n"
           << "Node has " << nodeInputTypes.size() << " inputs\n"
           << "Total number of arguments should be " << signatureParamTypes.size() + nodeInputTypes.size() << "\n"
           << "Body has " << bodyArgumentTypes.size() << " arguments. ";
  }

  for (size_t sigI = 0, argI = 0;
       sigI < signatureParamTypes.size(); ++sigI, ++argI) {
    if (signatureParamTypes[sigI] != bodyArgumentTypes[argI]) {
      return emitOpError("Mismatched types between lambda signature and body "
                         "arguments. ")
             << "Signature argument #" << sigI << " has type "
             << signatureParamTypes[sigI] << ". Body argument #" << argI
             << " has type " << bodyArgumentTypes[argI];
    }
  }

  for (size_t inI = 0, argI=signatureParamTypes.size(); inI < nodeInputTypes.size(); ++inI, ++argI) {
    if (nodeInputTypes[inI] != bodyArgumentTypes[argI]) {
      return emitOpError("Mismatched types in lambda node body arguments. "
                         "First arguments "
                         "should match node inputs. "
                         "Offending argument: #")
             << argI << " Expected " << nodeInputTypes[inI] << ", got "
             << bodyArgumentTypes[argI];
    }
  }

  return LogicalResult::success();
}

/**
 * Lambda node region terminator.
 * Verifies the following attributes:
 * - Number and types of operands when compared to function signature
 *  - Given a function signature with n results of types T1, T2, ..., Tn, the lambda node
 *    terminator should have n operands of types T1, T2, ..., Tn.
 */
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
        "Result type is invalid. Expected an instance of LambdaRefType.");
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

// CallableOpInterface methods for lambda node

mlir::Region* LambdaNode::getCallableRegion() {
  return &this->getRegion();
}

llvm::ArrayRef<mlir::Type> LambdaNode::getCallableResults() {
  auto type = this->getResult().getType().dyn_cast_or_null<LambdaRefType>();
  assert(type && "LambdaNode has invalid result type");
  return type.getReturnTypes();
}

/*
* Apply node verifier
* Verifies the following attributes:
* - Number and types of operands when compared to function signature
*  - Given a function signature with n operands of types T1, T2, ..., Tn, the apply node
*    should have n operands of types T1, T2, ..., Tn.
* - Number and types of outputs when compared to the signature of the lambda node
*  - Given a lambda signature with n results of types T1, T2, ..., Tn, the apply node
*    should have n outputs of types T1, T2, ..., Tn.
*/
LogicalResult ApplyNode::verify() {
  auto lambdaType = this->getLambda().getType();
  auto paramTypes = this->getParameters().getTypes();
  auto resultTypes = this->getResults().getTypes();

  if (lambdaType.getParameterTypes().size() != paramTypes.size()) {
    return this->emitOpError(" has the wrong number of parameters.")
    << " Lambda expects " << lambdaType.getParameterTypes().size()
    << " but " << paramTypes.size() << " were given.";
  }

  if (lambdaType.getParameterTypes().size() != paramTypes.size()) {
    return this->emitOpError(" has the wrong number of result types.")
    << " Lambda provides " << lambdaType.getParameterTypes().size()
    << " but " << paramTypes.size() << " were specified.";
  }

  size_t typeIndex = 0;
  for (auto [lambdaParam, nodeParam] : zip(lambdaType.getParameterTypes(), paramTypes)) {
    if (lambdaParam != nodeParam) {
      return emitOpError(" has mismatched parameter types.")
      << " Offending parameter: #" << typeIndex << "."
      << " Lambda expected " << lambdaParam << ", but got " << nodeParam;
    }
    ++ typeIndex;
  }

  typeIndex = 0;
  for (auto [lambdaResult, nodeResult] : zip(lambdaType.getReturnTypes(), resultTypes)) {
    if (lambdaResult != nodeResult) {
      return emitOpError(" has mismatched result types.")
      << " Offending result: #" << typeIndex << "."
      << " Lambda expected " << lambdaResult << ", but got " << nodeResult;
    }
    ++ typeIndex;
  }

  return LogicalResult::success();
}

// CallableOpInterface methods for apply node
mlir::CallInterfaceCallable ApplyNode::getCallableForCallee() {
  return getLambda();
}

mlir::Operation::operand_range ApplyNode::getArgOperands() {
  return this->getOperands().drop_front();
}

/**
 * Theta node verifier.
 * Verifies the following attributes:
 * - There should be an equal number of inputs, outputs, and region arguments, and they
 *   should all have the same types appear in the same order.
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

/* 
* Theta result verifier.
* Verifies the following attributes:
* - Number and types of non-predicate operands match the outputs of the parent
*   theta node.
*/
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

/*
 * Implements the MemoryEffectInterface for the theta node.
 * If the node has a memory state input or output, it currently
 * assumes that it has all memory effects. This can probably
 * be improved by analyzing the ops in the region. 
 * 
 * This interface could probably be generically implemented
 * for all structural nodes.
 * 
 * TODO: Consider baking specific side-effects into the
 *       state type.
*/
void ThetaNode::getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects) {
  for (auto input : this->getInputs()) {
    if (input.getType().isa<rvsdg::MemStateEdgeType>()) {
      effects.push_back(mlir::MemoryEffects::Write::get());
      effects.push_back(mlir::MemoryEffects::Read::get());
      effects.push_back(mlir::MemoryEffects::Allocate::get());
      effects.push_back(mlir::MemoryEffects::Free::get());
      return;
    }
  }
  for (auto output : this->getOutputs()) {
    if (output.getType().isa<rvsdg::MemStateEdgeType>()) {
      effects.push_back(mlir::MemoryEffects::Write::get());
      effects.push_back(mlir::MemoryEffects::Read::get());
      effects.push_back(mlir::MemoryEffects::Allocate::get());
      effects.push_back(mlir::MemoryEffects::Free::get());
      return;
    }
  }
}


/**
 * Phi node verifier.
 * Verifies the following attributes:
 * - Given n inputs of type I1, I2, ..., Tn, and k outputs of type O1, O2, ..., Ok,
 *   the region should have n + k arguments of type I1, I2, ..., Tn, O1, O2, ..., Ok.
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

/**
 * Phi result verifier.
 * Verifies the following attributes:
 * - Number and types of operands match the outputs of the parent phi node.
 */
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
  return LogicalResult::success();
}

/**
 * Delta node verifier.
 * Verifies the following attributes:
 * - Number and types of inputs match the region arguments.
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

/*
* Delta result verifier.
* Verifies the following attributes:
* - Number and types of operands match the output of the parent delta node.
*   - Delta output is a typed pointer which references the value given to the operand,
*     so the operand type should match the element type of the output.
*/
LogicalResult DeltaResult::verify() {
  auto parent = dyn_cast<DeltaNode>((*this)->getParentOp());
  if (parent == NULL) {
    return emitOpError("DeltaResult has no parent of type DeltaNode.");
  }
  auto resultType = this->getOperand().getType();
  auto outputType = parent.getOutput().getType();
  mlir::Type outputElementType;
  if (auto rvsdgPtrType = outputType.dyn_cast_or_null<RVSDGPointerType>()) {
    outputElementType = rvsdgPtrType.getElementType();
  } else if (auto llvmPtrType = outputType.dyn_cast_or_null<mlir::LLVM::LLVMPointerType>()) {
    outputElementType = llvmPtrType.getElementType(); 
  }
  if (resultType != outputElementType) {
    return emitOpError("Type mismatch between DeltaResult and DeltaNode output.")
           << " DeltaResult type: " << resultType
           << " DeltaNode output element type: " << outputElementType;
  }
  return LogicalResult::success();
}

/**
 * Match operator verifier.
 * Verifies the following attributes:
 * - No duplicate inputs in the match rules
 * - For a match function that produces a control type with n options, the productions
 *   of a mapping rule is in the range [0, n-1]
 */
LogicalResult rvsdg::Match::verify() {
  auto mappingAttr = this->getMapping();
  auto nOptions = this->getOutput().getType().getNumOptions();

  std::unordered_map<int64_t, size_t> seenInputs;
  bool hasDefault = false;
  size_t ruleIndex = 0;
  for (auto opaqueAttr : mappingAttr) {
    if (auto matchRuleAttr = opaqueAttr.dyn_cast<MatchRuleAttr>()) {
      if (matchRuleAttr.isDefault()) {
        if (hasDefault) {
          return emitOpError("Match operator has more than one default rule in its mapping attribute.");
        } else {
          hasDefault = true;
        }
      }
      auto matchValues = matchRuleAttr.getValues();
      for (auto value : matchValues) {
        if (seenInputs.count(value) != 0) {
          return emitOpError(" has a duplicate input in its mapping attribute.")
          << " Input " << value
          << " in rule #" << ruleIndex
          << ". Previously seen in rule #" << seenInputs[value];
        }
        seenInputs.emplace(value, ruleIndex);
      }

      auto matchResult = matchRuleAttr.getIndex();
      if (matchResult >= nOptions) {
        return emitOpError(" has a result index that is out of bounds in its mapping attribute.")
        << " Result index: " << matchResult
        << " Number of options: " << nOptions;
      }
      ruleIndex++;
    } else {
      return emitOpError("Match operator has a non-MatchRuleAttr attribute in its mapping attribute.");
    }
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
