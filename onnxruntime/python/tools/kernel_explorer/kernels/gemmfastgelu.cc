// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/gemmfastgelu.h"

#include <pybind11/pybind11.h>
#include <type_traits>

#include "core/providers/rocm/tunable/gemmfastgelu_common.h"
#include "python/tools/kernel_explorer/kernels/gemmfastgelu_ck.h"

using namespace onnxruntime::rocm::tunable::blas;

namespace py = pybind11;

namespace onnxruntime {

void InitGemmFastGelu(py::module mod) {
  InitComposableKernelGemmFastGelu(mod);
}

}  // namespace onnxruntime
