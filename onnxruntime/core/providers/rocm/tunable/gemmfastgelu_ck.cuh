// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_add_add_fastgelu.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "core/providers/rocm/tunable/gemmfastgelu_common.h"


namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

template <typename T>
struct DataTypeAdaptor {
  using type = T;
};

template <>
struct DataTypeAdaptor<half> {
  using type = ck::half_t;
};

template <>
struct DataTypeAdaptor<BFloat16> {
  using type = ck::bhalf16_t;
};

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;
using AddAddFastGelu = ck::tensor_operation::element_wise::AddAddFastGelu;

template <typename T, typename ALayout, typename BLayout>
auto GetCKGemmFastGeluTypeStringAndOps() {
  using CKDataType = typename DataTypeAdaptor<T>::type;
  using DeviceGemmAddAddFastGelu = ck::tensor_operation::device::DeviceGemmMultipleD<
      ALayout, BLayout, ck::Tuple<Row, Row>, Row,
      CKDataType, CKDataType, ck::Tuple<CKDataType, CKDataType>, CKDataType,
      Nop, Nop, AddAddFastGelu>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemmAddAddFastGelu>;
  std::cout << "found " << InstanceFactory::GetInstances().size() << " instances" << std::endl;

  std::vector<std::pair<std::string, Op<GemmFastGeluParams<T>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = impl->GetTypeString();
    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemmaddaddfastgelu_op = [impl = std::move(impl), invoker = std::move(invoker)](const GemmFastGeluParams<T>* params) -> Status {
      auto one = ToHipType<T>::FromFloat(1.0f);
      auto zero = ToHipType<T>::FromFloat(0.0f);

      TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(
          params->alpha != one || params->beta != zero,
          impl->GetTypeString(), " only supports alpha == 1 and beta == 0", params->Signature());

      auto nop = Nop{};
      auto addaddfastgelu = AddAddFastGelu{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b,
                                           std::array<const void*, 2>{params->bias0, params->bias1},
                                           params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb,
                                           std::array<ck::index_t, 2>{0, 0},
                                           params->ldc,
                                           nop, nop, addaddfastgelu);
      TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                               impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->stream});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemmaddaddfastgelu_op)));
  }
  return ret;
}

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
