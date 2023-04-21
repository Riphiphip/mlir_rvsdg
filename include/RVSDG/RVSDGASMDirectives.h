#pragma once
#include "mlir/IR/OpImplementation.h"

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
void printTypedParamList(mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::OperandRange operands, mlir::TypeRange types);

mlir::ParseResult parseTypedParamList(mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands, llvm::SmallVectorImpl<mlir::Type> &types);

/**
 * @brief Prints a list of regions prefixed with a list of region arguments
 * and their types
 *
 * @param p Assembly printer
 * @param op Operation which we are printing
 * @param regions Regions of the operation
 **/
void printRVSDGRegion(mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::Region &region);

void printRVSDGRegions(mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::MutableArrayRef<mlir::Region> regions);

mlir::ParseResult parseRVSDGRegion(mlir::OpAsmParser &parser, mlir::Region &region);

mlir::ParseResult parseRVSDGRegions(mlir::OpAsmParser &parser, llvm::SmallVectorImpl<std::unique_ptr<mlir::Region>> &regions);
