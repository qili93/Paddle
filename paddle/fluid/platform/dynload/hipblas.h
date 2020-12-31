/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag hipblas_dso_flag;
extern void *hipblas_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_HIPBLAS_WRAP(__name)                              \
  struct DynLoad__##__name {                                                  \
    template <typename... Args>                                               \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {   \
      using hipblas_func =                                                    \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);           \
      std::call_once(hipblas_dso_flag, []() {                                 \
        hipblas_dso_handle = paddle::platform::dynload::GetCublasDsoHandle(); \
      });                                                                     \
      static void *p_##__name = dlsym(hipblas_dso_handle, #__name);           \
      return reinterpret_cast<hipblas_func>(p_##__name)(args...);             \
    }                                                                         \
  };                                                                          \
  extern DynLoad__##__name __name

#define HIPBLAS_BLAS_ROUTINE_EACH(__macro)            \
  __macro(hipblasSaxpy);                \
  __macro(hipblasDaxpy);                \
  __macro(hipblasCaxpy);                \
  __macro(hipblasZaxpy);                \
  __macro(hipblasSscal);                \
  __macro(hipblasDscal);                \
  __macro(hipblasScopy);                \
  __macro(hipblasDcopy);                \
  __macro(hipblasSgemv);                \
  __macro(hipblasDgemv);                \
  __macro(hipblasCgemv);                \
  __macro(hipblasZgemv);                \
  __macro(hipblasSgemm);                \
  __macro(hipblasDgemm);                \
  __macro(hipblasZgemm);                \
  __macro(hipblasHgemm);                 \
  __macro(hipblasSgeam);                   \
  __macro(hipblasDgeam);                   \
  __macro(hipblasStrsm);                \
  __macro(hipblasDtrsm);                \
  __macro(hipblasCreate);               \
  __macro(hipblasDestroy);              \
  __macro(hipblasSetStream);            \
  __macro(hipblasSetPointerMode);       \
  __macro(hipblasGetPointerMode);       \
  __macro(hipblasSgemmBatched);            \
  __macro(hipblasDgemmBatched);            \
  __macro(hipblasCgemmBatched);            \
  __macro(hipblasZgemmBatched);

// HIP platform not supported, refer to the following url:
// https://github.com/ROCm-Developer-Tools/HIP/blob/roc-3.5.x/docs/markdown/CUBLAS_API_supported_by_HIP.md
// __macro(cublasSgemmEx); 
// __macro(cublasSgetrfBatched);
// __macro(cublasSgetriBatched);
// __macro(cublasDgetrfBatched);
// __macro(cublasDgetriBatched);
// __macro(cublasSmatinvBatched);
// __macro(cublasDmatinvBatched);

HIPBLAS_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HIPBLAS_WRAP)

#define HIPBLAS_BLAS_ROUTINE_EACH_R2(__macro) \
  __macro(hipblasGemmEx);                     \
  __macro(hipblasSgemmStridedBatched);        \
  __macro(hipblasDgemmStridedBatched);        \
  __macro(hipblasCgemmStridedBatched);        \
  __macro(hipblasZgemmStridedBatched);        \
  __macro(hipblasHgemmStridedBatched);

HIPBLAS_BLAS_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_HIPBLAS_WRAP)

// HIP platform not supported, refer to the following url:
// https://github.com/ROCm-Developer-Tools/HIP/blob/roc-3.5.x/docs/markdown/CUBLAS_API_supported_by_HIP.md
// __macro(cublasSetMathMode);
// __macro(cublasGetMathMode);
// __macro(cublasGemmBatchedEx);
// __macro(cublasGemmStridedBatchedEx);

#undef DECLARE_DYNAMIC_LOAD_HIPBLAS_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
