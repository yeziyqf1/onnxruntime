// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

#include "core/common/common.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
Status LaunchGemmFastGeluKernel(bool is_tuning,
                                hipStream_t stream,
                                rocblas_handle handle,
                                bool transa,
                                bool transb,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                const T alpha,
                                const T* a,
                                int64_t lda,
                                const T* b,
                                int64_t ldb,
                                const T* bias,
                                T* c,
                                int64_t ldc,
                                const T beta);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
