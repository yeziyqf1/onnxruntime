// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {

// We don't assume the implementation is row-majored or column-majored. But for testing convenience, we assume all
// our wrappers have row-majored convention, since it is the native layout to numpy and pytorch.
template <typename T>
struct GemmFastGeluParams : tunable::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k);
  }

  rocblas_handle handle;
  BlasOp opa;
  BlasOp opb;
  int64_t m;
  int64_t n;
  int64_t k;
  T alpha;
  const T* a;
  int64_t lda;
  const T* b;
  int64_t ldb;
  const T* bias0;
  const T* bias1;
  T beta;
  T* c;
  int64_t ldc;
};

}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
