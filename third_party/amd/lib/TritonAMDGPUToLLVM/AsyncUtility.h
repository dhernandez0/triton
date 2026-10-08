#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_

namespace mlir::triton::AMD {
class TargetInfo;

// Finds the largest supported vecSize smaller than maxVecSize. Returns 0 if
// there is none
unsigned
fitToValidDirectToLdsVecSize(unsigned maxVecSize, unsigned elemBitwidth,
                             const triton::AMD::TargetInfo &targetInfo);

} // namespace mlir::triton::AMD

#endif
