#include <unordered_set>

#include "mlir/IR/Block.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "JLM/JLMOps.h"

using namespace mlir;
using namespace jlm;

/**
 * Memory operators
 */

/**
 * Load
 */
LogicalResult jlm::Load::verify() {
  if (auto llvmPtr = this->getPointer().getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (llvmPtr.isOpaque()) {
      return LogicalResult::success();
    }
  }
  mlir::Type pointerElementType;
  if (auto rvsdgPtrType = this->getPointer().getType().dyn_cast_or_null<rvsdg::RVSDGPointerType>()) {
    pointerElementType = rvsdgPtrType.getElementType();
  } else if (auto llvmPtrType = this->getPointer().getType().dyn_cast_or_null<mlir::LLVM::LLVMPointerType>()) {
    pointerElementType = llvmPtrType.getElementType(); 
  } else {
    return emitOpError(" has a pointer that is not a pointer type.");
  }

  auto outputType = this->getOutput().getType();
  if (pointerElementType != outputType) {
    return emitOpError(" has a type mismatch between pointer and output.")
           << " Pointer element type: " << pointerElementType
           << " Output type: " << outputType;
  }
  return LogicalResult::success();
}

/*
* Store
*/
LogicalResult jlm::Store::verify() {
  if (auto llvmPtr = this->getPointer().getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (llvmPtr.isOpaque()) {
      return LogicalResult::success();
    }
  }
  mlir::Type pointerElementType;
  if (auto rvsdgPtrType = this->getPointer().getType().dyn_cast_or_null<rvsdg::RVSDGPointerType>()) {
    pointerElementType = rvsdgPtrType.getElementType();
  } else if (auto llvmPtrType = this->getPointer().getType().dyn_cast_or_null<mlir::LLVM::LLVMPointerType>()) {
    pointerElementType = llvmPtrType.getElementType(); 
  } else {
    return emitOpError(" has a pointer that is not a pointer type.");
  }

  auto valueType = this->getValue().getType();
  if (pointerElementType != valueType) {
    return emitOpError(" has a type mismatch between pointer and value.")
           << " Pointer element type: " << pointerElementType
           << " Value type: " << valueType;
  }
  return LogicalResult::success();
}

/**
 * Alloca
 */
LogicalResult jlm::Alloca::verify() {
  auto outputType = this->getOutput().getType();
  if (auto llvmPtrType = outputType.dyn_cast_or_null<mlir::LLVM::LLVMPointerType>()) {
    if (llvmPtrType.isOpaque()) {
      return LogicalResult::success();
    }
  }

  mlir::Type elementType;
  if (auto rvsdgPtrType = outputType.dyn_cast_or_null<rvsdg::RVSDGPointerType>()) {
    elementType = rvsdgPtrType.getElementType();
  } else if (auto llvmPtrType = outputType.dyn_cast_or_null<mlir::LLVM::LLVMPointerType>()){
    elementType = llvmPtrType.getElementType();
  } else {
    return emitOpError(" has an output that is not a pointer type.");
  }
  auto valueType = this->getTypeAttr();
  if (elementType != valueType) {
    return emitOpError(" has a type mismatch between output pointer and allocated type.")
           << " Pointer element type: " << elementType
           << " Allocated type: " << valueType;
  }
  return LogicalResult::success();
}

/**
 * Auto generated sources
 */
#define GET_OP_CLASSES
#include "JLM/Ops.cpp.inc"

/**
 * Implement dialect method for registering Ops
 */
void mlir::jlm::JLMDialect::addJLMOps() {
  addOperations<
#define GET_OP_LIST
#include "JLM/Ops.cpp.inc"
      >();
}
