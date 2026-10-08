#include "AsyncUtility.h"

#include "TargetInfo.h"

namespace mlir::triton::AMD {

unsigned
fitToValidDirectToLdsVecSize(unsigned maxVecSize, unsigned elemBitwidth,
                             const triton::AMD::TargetInfo &targetInfo) {
  while (maxVecSize > 0 && !targetInfo.supportsDirectToLdsLoadBitWidth(
                               maxVecSize * elemBitwidth)) {
    maxVecSize /= 2;
  }
  return maxVecSize;
}

} // namespace mlir::triton::AMD
