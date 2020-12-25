//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/rocblas.h"
#include "paddle/fluid/platform/gpu_info.h"

DECLARE_bool(enable_cublas_tensor_op_math);

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CUBlas;

template <>
struct CUBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_sgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_saxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_sscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_scopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_sgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::rocblas_sgemm_strided_batched(args...));
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    PADDLE_THROW("Currently there are not rocblas_gemm_ex.");
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_dgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_daxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_dscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_dcopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_dgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::rocblas_dgemm_strided_batched(args...));
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    PADDLE_THROW("Currently there are not rocblas_dgemmEx.");
  }
};

template <>
struct CUBlas<platform::float16> {
  using float16 = platform::float16;

  static void GEMM(rocblas_handle handle, rocblas_operation transa,
                   rocblas_operation transb, int m, int n, int k,
                   const float16 *alpha, const float16 *A, int lda,
                   const float16 *B, int ldb, const float16 *beta, float16 *C,
                   int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::rocblas_hgemm(handle, transa, transb, m, n, k,
                                       reinterpret_cast<const rocblas_half*>(alpha),
                                       reinterpret_cast<const rocblas_half*>(A), lda,
                                       reinterpret_cast<const rocblas_half*>(B), ldb,
                                       reinterpret_cast<const rocblas_half*>(beta),
                                       reinterpret_cast<rocblas_half*>(C), ldc));
  }

  static void GEMM_STRIDED_BATCH(rocblas_handle handle,
                                 rocblas_operation transa,
                                 rocblas_operation transb, int m, int n, int k,
                                 const float16 *alpha, const float16 *A,
                                 int lda, long long int strideA,  // NOLINT
                                 const float16 *B,                // NOLINT
                                 int ldb, long long int strideB,  // NOLINT
                                 const float16 *beta, float16 *C, int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
//#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::rocblas_hgemm_strided_batched(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const rocblas_half*>(alpha),
        reinterpret_cast<const rocblas_half*>(A), lda, strideA,
        reinterpret_cast<const rocblas_half*>(B), ldb, strideB,
        reinterpret_cast<const rocblas_half*>(beta), reinterpret_cast<rocblas_half*>(C),
        ldc, strideC, batchCount));
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    PADDLE_THROW("Currently there are not rocblas_gemmEx.");
  }

};

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(CBLAS_TRANSPOSE transA,
                                             CBLAS_TRANSPOSE transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             const T *B, T beta, T *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA =
      (transA == CblasNoTrans) ? rocblas_operation_none : rocblas_operation_transpose;
  rocblas_operation cuTransB =
      (transB == CblasNoTrans) ? rocblas_operation_none : rocblas_operation_transpose;

    context_.CublasCall([&](rocblas_handle handle) {
      CUBlas<T>::GEMM(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A,
                      lda, &beta, C, N);
    });
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::float16 alpha, const platform::float16 *A,
    const platform::float16 *B, platform::float16 beta,
    platform::float16 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA =
      (transA == CblasNoTrans) ? rocblas_operation_none : rocblas_operation_transpose;
  rocblas_operation cuTransB =
      (transB == CblasNoTrans) ? rocblas_operation_none : rocblas_operation_transpose;

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<platform::float16>::GEMM(handle, cuTransB, cuTransA, N, M, K,
                                    &alpha, B, ldb, A, lda, &beta, C,
                                    N);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(bool transA, bool transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             int lda, const T *B, int ldb,
                                             T beta, T *C, int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  rocblas_operation cuTransA = transA ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation cuTransB = transB ? rocblas_operation_transpose : rocblas_operation_none;

    context_.CublasCall([&](rocblas_handle handle) {
      CUBlas<T>::GEMM(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A,
                      lda, &beta, C, ldc);
    });
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    bool transA, bool transB, int M, int N, int K, platform::float16 alpha,
    const platform::float16 *A, int lda, const platform::float16 *B, int ldb,
    platform::float16 beta, platform::float16 *C, int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  rocblas_operation cuTransA = transA ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation cuTransB = transB ? rocblas_operation_transpose : rocblas_operation_none;

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<platform::float16>::GEMM(handle, cuTransB, cuTransA, N, M, K, &alpha,
                                    B, ldb, A, lda, &beta, C, ldc);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::AXPY(int n, T alpha, const T *x,
                                             T *y) const {
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::AXPY(handle, n, &alpha, x, 1, y, 1);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::SCAL(int n, const T alpha, T *x) const {
  context_.CublasCall(
      [&](rocblas_handle handle) { CUBlas<T>::SCAL(handle, n, &alpha, x, 1); });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::VCOPY(int n, const T *x, T *y) const {
  context_.CublasCall(
      [&](rocblas_handle handle) { CUBlas<T>::VCOPY(handle, n, x, 1, y, 1); });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMV(bool trans_a, int M, int N,
                                             T alpha, const T *A, const T *B,
                                             T beta, T *C) const {
  rocblas_operation cuTransA = !trans_a ? rocblas_operation_transpose : rocblas_operation_none;

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GEMV(handle, cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    T alpha, const T *A, const T *B, T beta, T *C, int batchCount,
    int64_t strideA, int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  rocblas_operation cuTransA =
      (transA == CblasNoTrans) ? rocblas_operation_none : rocblas_operation_transpose;
  rocblas_operation cuTransB =
      (transB == CblasNoTrans) ? rocblas_operation_none : rocblas_operation_transpose;
  const int64_t strideC = M * N;

    context_.CublasCall([&](rocblas_handle handle) {
      CUBlas<T>::GEMM_STRIDED_BATCH(handle, cuTransB, cuTransA, N, M, K, &alpha,
                                    B, ldb, strideB, A, lda, strideA, &beta, C,
                                    ldc, strideC, batchCount);
    });
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
