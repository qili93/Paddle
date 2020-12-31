//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/dynload/hipblas.h"
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
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasSgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasSaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasSscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasScopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasSgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::hipblasSgemmStridedBatched(args...));
  }

  // HIP not supportted, refer to the doc here:
  // https://github.com/ROCm-Developer-Tools/HIP/blob/roc-3.5.x/docs/markdown/CUBLAS_API_supported_by_HIP.md
  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasSgemmEx is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasStrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasSgetrfBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasSgetriBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasSmatinvBatched is not supported on HIP platform."));
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasDgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasDaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasDscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasDcopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasDgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::hipblasDgemmStridedBatched(args...));
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Currently there are not cublasDgemmEx."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasDtrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasDgetrfBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasDgetriBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasDmatinvBatched is not supported on HIP platform."));
  }
};

template <>
struct CUBlas<platform::float16> {
  using float16 = platform::float16;

  static void GEMM(hipblasHandle_t handle, hipblasOperation_t transa,
                   hipblasOperation_t transb, int m, int n, int k,
                   const float16 *alpha, const float16 *A, int lda,
                   const float16 *B, int ldb, const float16 *beta, float16 *C,
                   int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::hipblasHgemm(handle, transa, transb, m, n, k,
                                       reinterpret_cast<const hipblasHalf*>(alpha),
                                       reinterpret_cast<const hipblasHalf*>(A), lda,
                                       reinterpret_cast<const hipblasHalf*>(B), ldb,
                                       reinterpret_cast<const hipblasHalf*>(beta),
                                       reinterpret_cast<hipblasHalf*>(C), ldc));
  }

  static void GEMM_STRIDED_BATCH(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb, int m, int n, int k,
                                 const float16 *alpha, const float16 *A,
                                 int lda, long long int strideA,  // NOLINT
                                 const float16 *B,                // NOLINT
                                 int ldb, long long int strideB,  // NOLINT
                                 const float16 *beta, float16 *C, int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasHgemmStridedBatched(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const hipblasHalf*>(alpha),
        reinterpret_cast<const hipblasHalf*>(A), lda, strideA,
        reinterpret_cast<const hipblasHalf*>(B), ldb, strideB,
        reinterpret_cast<const hipblasHalf*>(beta), reinterpret_cast<hipblasHalf*>(C),
        ldc, strideC, batchCount));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(platform::CUDADeviceContext *dev_ctx,
                      hipblasOperation_t transa, hipblasOperation_t transb, int m,
                      int n, int k, const void *alpha, const void *A,
                      hipblasDatatype_t Atype, int lda, const void *B,
                      hipblasDatatype_t Btype, int ldb, const void *beta, void *C,
                      hipblasDatatype_t Ctype, int ldc,
                      hipblasDatatype_t computeType) {
    hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](hipblasHandle_t handle) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasGemmEx(
          handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
          beta, C, Ctype, ldc, computeType, algo));
    });
  }
};

template <>
struct CUBlas<platform::complex64> {
  using complex64 = platform::complex64;

  static void GEMV(hipblasHandle_t handle, hipblasOperation_t transa, int m,
                   int n, const complex64 *alpha, const complex64 *A, int lda,
                   const complex64 *B, int ldb, const complex64 *beta,
                   complex64 *C, int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasCgemv(
        handle, transa, m, n, reinterpret_cast<const hipblasComplex*>(alpha),
        reinterpret_cast<const hipblasComplex*>(A), lda,
        reinterpret_cast<const hipblasComplex*>(B), ldb,
        reinterpret_cast<const hipblasComplex*>(beta),
        reinterpret_cast<hipblasComplex*>(C), ldc));
  }

  static void AXPY(hipblasHandle_t handle, int n, const complex64 *alpha,
                   const complex64 *X, const int incX, complex64 *Y,
                   const int incY) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasCaxpy(
        handle, n, reinterpret_cast<const hipblasComplex*>(alpha),
        reinterpret_cast<const hipblasComplex*>(X), incX,
        reinterpret_cast<hipblasComplex*>(Y), incY));
  }

  static void GEMM_STRIDED_BATCH(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb, int m, int n, int k,
                                 const complex64 *alpha, const complex64 *A,
                                 int lda, long long int strideA,  // NOLINT
                                 const complex64 *B,              // NOLINT
                                 int ldb, long long int strideB,  // NOLINT
                                 const complex64 *beta, complex64 *C, int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasCgemmStridedBatched(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const hipblasComplex*>(alpha),
        reinterpret_cast<const hipblasComplex*>(A), lda, strideA,
        reinterpret_cast<const hipblasComplex*>(B), ldb, strideB,
        reinterpret_cast<const hipblasComplex*>(beta),
        reinterpret_cast<hipblasComplex*>(C), ldc, strideC, batchCount));
  }

  static void GEMM(hipblasHandle_t handle, hipblasOperation_t transa,
                   hipblasOperation_t transb, int m, int n, int k,
                   const complex64 *alpha, const complex64 *A, int lda,
                   const complex64 *B, int ldb, const complex64 *beta,
                   complex64 *C, int ldc) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasCgemm is not supported on HIP platform."));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(platform::CUDADeviceContext *dev_ctx,
                      hipblasOperation_t transa, hipblasOperation_t transb, int m,
                      int n, int k, const void *alpha, const void *A,
                      hipblasDatatype_t Atype, int lda, const void *B,
                      hipblasDatatype_t Btype, int ldb, const void *beta, void *C,
                      hipblasDatatype_t Ctype, int ldc,
                      hipblasDatatype_t computeType) {

    hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](hipblasHandle_t handle) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasGemmEx(
          handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
          beta, C, Ctype, ldc, computeType, algo));
    });
  }
};

template <>
struct CUBlas<platform::complex128> {
  using complex128 = platform::complex128;

  static void GEMV(hipblasHandle_t handle, hipblasOperation_t transa, int m,
                   int n, const complex128 *alpha, const complex128 *A, int lda,
                   const complex128 *B, int ldb, const complex128 *beta,
                   complex128 *C, int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasZgemv(
        handle, transa, m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha),
        reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
        reinterpret_cast<const hipblasDoubleComplex*>(B), ldb,
        reinterpret_cast<const hipblasDoubleComplex*>(beta),
        reinterpret_cast<hipblasDoubleComplex*>(C), ldc));
  }

  static void AXPY(hipblasHandle_t handle, int n, const complex128 *alpha,
                   const complex128 *X, const int incX, complex128 *Y,
                   const int incY) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasZaxpy(
        handle, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha),
        reinterpret_cast<const hipblasDoubleComplex*>(X), incX,
        reinterpret_cast<hipblasDoubleComplex*>(Y), incY));
  }

  static void GEMM_STRIDED_BATCH(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb, int m, int n, int k,
                                 const complex128 *alpha, const complex128 *A,
                                 int lda, long long int strideA,  // NOLINT
                                 const complex128 *B,             // NOLINT
                                 int ldb, long long int strideB,  // NOLINT
                                 const complex128 *beta, complex128 *C, int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasZgemmStridedBatched(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
        reinterpret_cast<const hipblasDoubleComplex*>(A), lda, strideA,
        reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, strideB,
        reinterpret_cast<const hipblasDoubleComplex*>(beta),
        reinterpret_cast<hipblasDoubleComplex*>(C), ldc, strideC, batchCount));
  }

  static void GEMM(hipblasHandle_t handle, hipblasOperation_t transa,
                   hipblasOperation_t transb, int m, int n, int k,
                   const complex128 *alpha, const complex128 *A, int lda,
                   const complex128 *B, int ldb, const complex128 *beta,
                   complex128 *C, int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasZgemm(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
        reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
        reinterpret_cast<const hipblasDoubleComplex*>(B), ldb,
        reinterpret_cast<const hipblasDoubleComplex*>(beta),
        reinterpret_cast<hipblasDoubleComplex*>(C), ldc));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(platform::CUDADeviceContext *dev_ctx,
                      hipblasOperation_t transa, hipblasOperation_t transb, int m,
                      int n, int k, const void *alpha, const void *A,
                      hipblasDatatype_t Atype, int lda, const void *B,
                      hipblasDatatype_t Btype, int ldb, const void *beta, void *C,
                      hipblasDatatype_t Ctype, int ldc,
                      hipblasDatatype_t computeType) {
    hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](hipblasHandle_t handle) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipblasGemmEx(
          handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
          beta, C, Ctype, ldc, computeType, algo));
    });
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
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (transB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  context_.CublasCall([&](hipblasHandle_t handle) {
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
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (transB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(), 53,
      platform::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  auto &cuda_ctx = const_cast<platform::CUDADeviceContext &>(context_);
  CUBlas<platform::float16>::GEMM_EX(
      &cuda_ctx, cuTransB, cuTransA, N, M, K, &h_alpha, B, HIPBLAS_R_16F, ldb, A,
      HIPBLAS_R_16F, lda, &h_beta, C, HIPBLAS_R_16F, N, HIPBLAS_R_32F);
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::complex64 alpha, const platform::complex64 *A,
    const platform::complex64 *B, platform::complex64 beta,
    platform::complex64 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (transB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(), 53,
      platform::errors::InvalidArgument(
          "cublas complex64 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  thrust::complex<float> c_alpha =
      thrust::complex<float>(alpha.real, alpha.imag);
  thrust::complex<float> c_beta = thrust::complex<float>(beta.real, beta.imag);

  auto &cuda_ctx = const_cast<platform::CUDADeviceContext &>(context_);
  CUBlas<platform::complex64>::GEMM_EX(
      &cuda_ctx, cuTransB, cuTransA, N, M, K, &c_alpha, B, HIPBLAS_C_32F, ldb, A,
      HIPBLAS_C_32F, lda, &c_beta, C, HIPBLAS_C_32F, N, HIPBLAS_C_32F);
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::complex128 alpha, const platform::complex128 *A,
    const platform::complex128 *B, platform::complex128 beta,
    platform::complex128 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (transB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(), 53,
      platform::errors::InvalidArgument(
          "cublas complex128 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  thrust::complex<double> c_alpha =
      thrust::complex<double>(alpha.real, alpha.imag);
  thrust::complex<double> c_beta =
      thrust::complex<double>(beta.real, beta.imag);

  auto &cuda_ctx = const_cast<platform::CUDADeviceContext &>(context_);
  CUBlas<platform::complex128>::GEMM_EX(
      &cuda_ctx, cuTransB, cuTransA, N, M, K, &c_alpha, B, HIPBLAS_C_64F, ldb, A,
      HIPBLAS_C_64F, lda, &c_beta, C, HIPBLAS_C_64F, N, HIPBLAS_C_64F);
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(bool transA, bool transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             int lda, const T *B, int ldb,
                                             T beta, T *C, int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  hipblasOperation_t cuTransA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t cuTransB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  context_.CublasCall([&](hipblasHandle_t handle) {
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
  hipblasOperation_t cuTransA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t cuTransB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  context_.CublasCall([&](hipblasHandle_t handle) {
    CUBlas<platform::float16>::GEMM(handle, cuTransB, cuTransA, N, M, K, &alpha,
                                    B, ldb, A, lda, &beta, C, ldc);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::AXPY(int n, T alpha, const T *x,
                                             T *y) const {
  context_.CublasCall([&](hipblasHandle_t handle) {
    CUBlas<T>::AXPY(handle, n, &alpha, x, 1, y, 1);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::SCAL(int n, const T alpha, T *x) const {
  context_.CublasCall(
      [&](hipblasHandle_t handle) { CUBlas<T>::SCAL(handle, n, &alpha, x, 1); });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::VCOPY(int n, const T *x, T *y) const {
  context_.CublasCall(
      [&](hipblasHandle_t handle) { CUBlas<T>::VCOPY(handle, n, x, 1, y, 1); });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMV(bool trans_a, int M, int N,
                                             T alpha, const T *A, const T *B,
                                             T beta, T *C) const {
  hipblasOperation_t cuTransA = !trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  context_.CublasCall([&](hipblasHandle_t handle) {
    CUBlas<T>::GEMV(handle, cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1);
  });
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMV(
    bool trans_a, int M, int N, platform::float16 alpha,
    const platform::float16 *A, const platform::float16 *B,
    platform::float16 beta, platform::float16 *C) const {
  // Because cublas doesn't support half gemv, we use cublasHgemm to achieve it.
  if (trans_a) {
    this->template GEMM<platform::float16>(CblasNoTrans, CblasNoTrans, 1, N, M,
                                           alpha, B, A, beta, C);
  } else {
    this->template GEMM<platform::float16>(CblasNoTrans, CblasNoTrans, M, 1, N,
                                           alpha, A, B, beta, C);
  }
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
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (transB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  const int64_t strideC = M * N;
  context_.CublasCall([&](hipblasHandle_t handle) {
    CUBlas<T>::GEMM_STRIDED_BATCH(handle, cuTransB, cuTransA, N, M, K, &alpha,
                                  B, ldb, strideB, A, lda, strideA, &beta, C,
                                  ldc, strideC, batchCount);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    T alpha, const T **A, const T **B, T beta, T **C, int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<T>(transA, transB, M, N, K, alpha, A[k], B[k], beta,
                           C[k]);
  }
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::float16 alpha, const platform::float16 **A,
    const platform::float16 **B, platform::float16 beta, platform::float16 **C,
    int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<platform::float16>(transA, transB, M, N, K, alpha, A[k],
                                           B[k], beta, C[k]);
  }
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::TRSM(CBLAS_SIDE side, CBLAS_UPLO uplo,
                                             CBLAS_TRANSPOSE transA,
                                             CBLAS_DIAG diag, int M, int N,
                                             T alpha, const T *A, int lda, T *B,
                                             int ldb) const {
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  hipblasSideMode_t cuSide =
      (side == CblasLeft) ? HIPBLAS_SIDE_RIGHT : HIPBLAS_SIDE_LEFT;
  hipblasFillMode_t cuUplo =
      (uplo == CblasLower) ? HIPBLAS_FILL_MODE_UPPER : HIPBLAS_FILL_MODE_LOWER;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasDiagType_t cuDiag =
      (diag == CblasUnit) ? HIPBLAS_DIAG_UNIT : HIPBLAS_DIAG_NON_UNIT;

  context_.CublasCall([&](hipblasHandle_t handle) {
    CUBlas<T>::TRSM(handle, cuSide, cuUplo, cuTransA, cuDiag, N, M, &alpha, A,
                    lda, B, ldb);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGETRF(int n, T **a, int *ipiv,
                                                     int *info,
                                                     int batch_size) const {
  context_.CublasCall([&](hipblasHandle_t handle) {
    CUBlas<T>::GETRF_BATCH(handle, n, a, n, ipiv, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGETRI(int n, const T **a,
                                                     const int *ipiv, T **a_inv,
                                                     int *info,
                                                     int batch_size) const {
  PADDLE_ENFORCE_NE(
      a_inv, a,
      platform::errors::InvalidArgument(
          "cuBLAS fuction 'cublas<S/D>getrfBatched' cannot be executed "
          "in-place. The memory space of output matrix (address: %p) cannot "
          "overlap memory space of input matrix (address: %p).",
          a_inv, a));
  context_.CublasCall([&](hipblasHandle_t handle) {
    CUBlas<T>::GETRI_BATCH(handle, n, a, n, ipiv, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedMatInv(int n, const T **a,
                                                      T **a_inv, int *info,
                                                      int batch_size) const {
  context_.CublasCall([&](hipblasHandle_t handle) {
    CUBlas<T>::MATINV_BATCH(handle, n, a, n, a_inv, n, info, batch_size);
  });
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
