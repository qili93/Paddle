/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#endif
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

#ifdef __HIPCC__
#define KERNEL_PRINT(__FORMAT, ...)              \
        printf("%03d: [tid.x=<%lu> tid.y=<%lu> bid.x=<%lu> bid.y=<%lu>]: " __FORMAT "\n", \
        __LINE__, hipThreadIdx_x, hipThreadIdx_y, hipBlockIdx_x, hipBlockIdx_y, ##__VA_ARGS__);
#else
#define KERNEL_PRINT(__FORMAT, ...)              \
        printf("%03d: [tid.x=<%lu> tid.y=<%lu> bid.x=<%lu> bid.y=<%lu>]: " __FORMAT "\n", \
        __LINE__, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ##__VA_ARGS__);
#endif

using Tensor = framework::Tensor;

namespace {
template <typename T>
__global__ void CrossEntropyGrad(T* logit_grad, const int64_t* labels,
                                 const int n, const int d, const int remain,
                                 const int ignore_index) {
  CUDA_KERNEL_LOOP(index, n * remain) {
    int idx_n = index / remain;
    int idx_remain = index % remain;
    int tmp = labels[index];
    if (ignore_index != tmp) {
      int idx = idx_n * d + tmp * remain + idx_remain;
      logit_grad[idx] -= static_cast<T>(1.);
    }
  }
}

template <typename T>
__global__ void Scale(T* logit_grad, const T* loss_grad, const int num,
                      const int d, const int remain, const int64_t* labels,
                      const int ignore_index) {
  CUDA_KERNEL_LOOP(index, num) {
    int idx_n = index / d;
    int idx_remain = index % remain;
    int idx_lbl = idx_n * remain + idx_remain;
    if (labels[idx_lbl] == ignore_index) {
      logit_grad[index] = static_cast<T>(0.);
    } else {
      logit_grad[index] *= loss_grad[idx_lbl];
    }
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* logit_grad,
                                               const T* loss_grad,
                                               const T* labels, const int n,
                                               const int d, const int remain) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int idx_n = ids / d;
    int idx_remain = ids % remain;
    int idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (logit_grad[ids] - labels[ids]);
  }
}

}  // namespace

static __device__ __forceinline__ platform::float16 exp_on_device(
    platform::float16 x) {
  return ::Eigen::numext::exp(x);
}
static __device__ __forceinline__ float exp_on_device(float x) {
  return expf(x);
}
static __device__ __forceinline__ double exp_on_device(double x) {
  return exp(x);
}
static __device__ __forceinline__ platform::float16 log_on_device(
    platform::float16 x) {
  return math::TolerableValue<platform::float16>()(::Eigen::numext::log(x));
}
static __device__ __forceinline__ float log_on_device(float x) {
  return math::TolerableValue<float>()(logf(x));
}
static __device__ __forceinline__ double log_on_device(double x) {
  return math::TolerableValue<double>()(log(x));
}

/** In the following codes, 3 CUDA kernels are implemented to calculate softmax
 * and loss **/
/*
  Supposing the x is `logits` and y is `labels`, the equations are as
followings:
  cross\_entropy_i = \sum_{j}[- y_i_j * log({e^{x_i_j}/\sum_{j}e^{x_i_j}})]
        = \sum_{j}[- y_i_j * log({e^{x_i_j - max_i}/\sum_{j}e^{x_i_j-max_i}})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - log\sum_{j}e^{x_i_j - max_i})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - logDiffMaxSum_i)]
        = \sum_{j}(-y_i_j * tmp_i_j)
  softmax_i_j = e^{tmp_i_j}
where:
  max_i = \max_{j}{x_i_j}
  logDiffMaxSum_i = log\sum_{j}e^{x_i_j - max_i}
  tmp_i_j = x_i_j - max_i - logDiffMaxSum_i
Therefore, the calculation can be separated into 3 steps:
Step 1: row-wise operation to calculate max_i
Step 2: row-wise operation to calculate logDiffMaxSum_i
Step 3: calculate tmp_i_j, and finally get softmax_i_j and cross\_entropy_i
To save memory, we can share memory among max_i, logDiffMaxSum_i and
cross\_entropy_i.
In this way, the 3 steps should be changed to:
Step 1 (RowReductionForMax): row-wise operation to calculate max_i
Step 2 (RowReductionForDiffMaxSum): calculate immediate result of softmax'_i_j =
x_i_j - max_i, and row-wise operation to calculate logDiffMaxSum_i
Step 3 (RowReductionForSoftmaxAndCrossEntropy): calculate tmp_i_j = softmax'_i_j
- logDiffMaxSum_i, and finally get softmax_i_j and cross\_entropy_i
*/

// There are 3 kinds of reduce algorithms in cub:
// BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
// BLOCK_REDUCE_RAKING
// BLOCK_REDUCE_WARP_REDUCTIONS (default)
template <typename T, int BlockDim>
#ifdef __HIPCC__
using BlockReduce = hipcub::BlockReduce<T, BlockDim /*, hipcub::BLOCK_REDUCE_WARP_REDUCTIONS*/>;
#else
using BlockReduce = cub::BlockReduce<T, BlockDim /*, cub::BLOCK_REDUCE_WARP_REDUCTIONS*/>;
#endif

template <typename T, int BlockDim>
using BlockReduceTempStorage = typename BlockReduce<T, BlockDim>::TempStorage;

// Make sure that BlockDim <= axis_dim
// This kernel is used to calculate the max element of each row
template <typename T, int BlockDim>
static __global__ void RowReductionForMax(const T* logits_data, T* max_data,
                                          int d, int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits_data view as [n, axis_dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  // KERNEL_PRINT("remain=%d idx_n=%d idx_remain=%d beg_idx=%d end_idx=%d", 
                // remain, idx_n, idx_remain, beg_idx, end_idx);

  int step = BlockDim * remain;
  T cur_max = logits_data[beg_idx];
  // KERNEL_PRINT("beg_idx=%d logits_data[beg_idx]=%f cur_max=%f", 
                // beg_idx, static_cast<float>(logits_data[beg_idx]), static_cast<float>(cur_max));
  beg_idx += step;
  while (beg_idx < end_idx) {
    if (cur_max < logits_data[beg_idx]) {
      cur_max = logits_data[beg_idx];
    }
    beg_idx += step;
  }
#ifdef __HIPCC__
  cur_max = BlockReduce<T, BlockDim>(temp_storage).Reduce(cur_max, hipcub::Max());
#else
  cur_max = BlockReduce<T, BlockDim>(temp_storage).Reduce(cur_max, cub::Max());
#endif

  if (threadIdx.x == 0) max_data[blockIdx.x] = cur_max;
  //KERNEL_PRINT("==0== max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));

// #ifdef __HIPCC__
//   // Note(qili93): max_data should assign to correct value in all threads
//   // otherwise max_data value will be changed in other threads in HIPCC
//   // KERNEL_PRINT("cur_max=%f", static_cast<float>(cur_max));
//   if (threadIdx.x == 0) max_data[blockIdx.x] = cur_max;
//   // KERNEL_PRINT("max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));
// #else
//   if (threadIdx.x == 0) max_data[blockIdx.x] = cur_max;
// #endif
  // KERNEL_PRINT("==0== max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));

  // __syncthreads();

  // KERNEL_PRINT("==1== max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));
}

template <typename T, int BlockDim>
static __global__ void RowReductionForSum(const T* logits_data,
                                                 T* max_data, T* softmax, int d,
                                                 int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax data view as [n, axis_dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  auto block_max = max_data[blockIdx.x];
  int step = BlockDim * remain;

  // In numeric stable mode softmax_with_loss, we calc loss with
  // tmp_i_j = x_i_j - max_i - logDiffMaxSum_i, instead of
  // log(exp(x_i_j - max_i)/DiffMaxSum_i). Therefore, log(0) will not occur.
  // Also we calc softmax_i_j = e^{tmp_i_j}, the maximum and minimum value will
  // be 1.0 and 0.0, represent prob is 1.0 and 0.0.
  // So there is no need to clip on shift_softmax.
  softmax[beg_idx] = logits_data[beg_idx] - block_max;
  T diff_max_sum = exp_on_device(softmax[beg_idx]);
  auto idx = beg_idx + step;
  while (idx < end_idx) {
    softmax[idx] = logits_data[idx] - block_max;
    diff_max_sum += exp_on_device(softmax[idx]);
    idx += step;
  }

#ifdef __HIPCC__
  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, hipcub::Sum());
#else
  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());
#endif

  //KERNEL_PRINT("==0== diff_max_sum=%f", static_cast<float>(diff_max_sum));
  //KERNEL_PRINT("==1== diff_max_sum=%f", static_cast<float>(log_on_device(diff_max_sum)));

  if (threadIdx.x == 0) max_data[blockIdx.x] = log_on_device(diff_max_sum);
  //KERNEL_PRINT("==2== max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));
}

// Make sure that BlockDim <= axis_dim
template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
static __global__ void RowReductionForDiff(const T* logits_data,
                                                 T* max_data, T* softmax, int d,
                                                 int axis_dim) {
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;
  int step = BlockDim * remain;

  //KERNEL_PRINT("==3== max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));     
  T diff_max_sum = max_data[blockIdx.x];
  //KERNEL_PRINT("==4== diff_max_sum=%f", static_cast<float>(diff_max_sum));
  softmax[beg_idx] -= diff_max_sum;
  beg_idx += step;
  while (beg_idx < end_idx) {
    softmax[beg_idx] -= diff_max_sum;
    beg_idx += step;
  }
  //KERNEL_PRINT("==5== softmax[beg_idx]=%f", static_cast<float>(softmax[beg_idx]));     

  // Note(zhiqiu): since different threads may use max_data[blockIdx.x] to
  // calculate diff_max_sum, __syncthreads() is needed here.
  __syncthreads();

  //KERNEL_PRINT("==6== softmax[beg_idx]=%f", static_cast<float>(softmax[beg_idx]));     
  if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
  //KERNEL_PRINT("==7== max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));  
}

// // Make sure that BlockDim <= axis_dim
// template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
// static __global__ void RowReductionForDiffMaxSum(const T* logits_data,
//                                                  T* max_data, T* softmax, int d,
//                                                  int axis_dim) {
//   __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

//   // logits, softmax data view as [n, axis_dim, remain]
//   // max_data view as [n, 1, remain]
//   // blockDim = n * remain, split blockIdx to idx_n and idx_remain
//   int remain = d / axis_dim;
//   int idx_n = blockIdx.x / remain;
//   int idx_remain = blockIdx.x % remain;
//   int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
//   int end_idx = (idx_n + 1) * d;

//   auto block_max = max_data[blockIdx.x];
//   int step = BlockDim * remain;

//   // KERNEL_PRINT("remain=%d idx_n=%d idx_remain=%d beg_idx=%d end_idx=%d step=%d", 
//                 // remain, idx_n, idx_remain, beg_idx, end_idx, step);

//   // In numeric stable mode softmax_with_loss, we calc loss with
//   // tmp_i_j = x_i_j - max_i - logDiffMaxSum_i, instead of
//   // log(exp(x_i_j - max_i)/DiffMaxSum_i). Therefore, log(0) will not occur.
//   // Also we calc softmax_i_j = e^{tmp_i_j}, the maximum and minimum value will
//   // be 1.0 and 0.0, represent prob is 1.0 and 0.0.
//   // So there is no need to clip on shift_softmax.
//   softmax[beg_idx] = logits_data[beg_idx] - block_max;
//   T diff_max_sum = exp_on_device(softmax[beg_idx]);
//   // KERNEL_PRINT("beg_idx=%d logits_data[beg_idx]=%f block_max=%f softmax[beg_idx]=%f diff_max_sum=%f", 
//                 // beg_idx, static_cast<float>(logits_data[beg_idx]),
//                 // static_cast<float>(block_max),
//                 // static_cast<float>(softmax[beg_idx]),
//                 // static_cast<float>(diff_max_sum));
//   auto idx = beg_idx + step;
//   while (idx < end_idx) {
//     softmax[idx] = logits_data[idx] - block_max;
//     diff_max_sum += exp_on_device(softmax[idx]);
//     // KERNEL_PRINT("idx=%d logits_data[idx]=%f block_max=%f softmax[idx]=%f diff_max_sum=%f", 
//                   // idx, static_cast<float>(logits_data[idx]),
//                   // static_cast<float>(block_max),
//                   // static_cast<float>(softmax[idx]),
//                   // static_cast<float>(diff_max_sum));
//     idx += step;
//   }
//   // KERNEL_PRINT("==0== diff_max_sum=%f", static_cast<float>(diff_max_sum));
// #ifdef __HIPCC__
//   diff_max_sum =
//       BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, hipcub::Sum());
// #else
//   diff_max_sum =
//       BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());
// #endif

//   KERNEL_PRINT("==0== diff_max_sum=%f", static_cast<float>(diff_max_sum));
//   KERNEL_PRINT("==1== diff_max_sum=%f", static_cast<float>(log_on_device(diff_max_sum)));

// #ifdef __HIPCC__
//   if (threadIdx.x == 0) max_data[hipBlockIdx_x] = -1.0;
// #else
//   if (threadIdx.x == 0) max_data[blockIdx.x] = log_on_device(diff_max_sum);
// #endif

// #ifdef __HIPCC__
//   KERNEL_PRINT("==2== max_data[%lu]=%f", hipBlockIdx_x, static_cast<float>(max_data[hipBlockIdx_x]));
// #else
//   KERNEL_PRINT("==2== max_data[%d]=%f", blockIdx.x, static_cast<float>(max_data[blockIdx.x]));
// #endif

//   // NOTE: value of max data changed after sync thread !!!!!
//   __syncthreads();

// #ifdef __HIPCC__
//   KERNEL_PRINT("==3== max_data[%lu]=%f", hipBlockIdx_x, static_cast<float>(max_data[hipBlockIdx_x]));
// #else
//   KERNEL_PRINT("==3== max_data[%d]=%f", blockIdx.x, static_cast<float>(max_data[blockIdx.x]));
// #endif

//   if (!CalculateLogSoftmax) return;

//   __syncthreads();

// #ifdef __HIPCC__
//   KERNEL_PRINT("==4== max_data[%lu]=%f", hipBlockIdx_x, static_cast<float>(max_data[hipBlockIdx_x]));
// #else
//   KERNEL_PRINT("==4== max_data[%d]=%f", blockIdx.x, static_cast<float>(max_data[blockIdx.x]));
// #endif

// #ifdef __HIPCC__
//   diff_max_sum = max_data[hipBlockIdx_x];
// #else
//   diff_max_sum = max_data[blockIdx.x];
// #endif
//   // KERNEL_PRINT("diff_max_sum=%f", static_cast<float>(diff_max_sum));
//   softmax[beg_idx] -= diff_max_sum;
//   // KERNEL_PRINT("max_data[blockIdx.x]=%f diff_max_sum=%f beg_idx=%d softmax[beg_idx]=%f",
//                 // static_cast<float>(max_data[blockIdx.x]),
//                 // static_cast<float>(diff_max_sum),
//                 // beg_idx,
//                 // static_cast<float>(softmax[beg_idx]));

//   beg_idx += step;
//   while (beg_idx < end_idx) {
//     softmax[beg_idx] -= diff_max_sum;
//     // KERNEL_PRINT("beg_idx=%d end_idx=%d step=%d diff_max_sum=%f softmax[beg_idx]=%f", 
//                   // beg_idx, end_idx, step,
//                   // static_cast<float>(diff_max_sum),
//                   // static_cast<float>(softmax[beg_idx]));
//     beg_idx += step;
//   }

//   // Note(zhiqiu): since different threads may use max_data[blockIdx.x] to
//   // calculate diff_max_sum, __syncthreads() is needed here.
//   __syncthreads();
// #ifdef __HIPCC__
//   // KERNEL_PRINT("max_data[%lu]=%f", hipBlockIdx_x, static_cast<float>(max_data[blockIdx.x]));
// #else
//   // KERNEL_PRINT("max_data[%d]=%f", blockIdx.x, static_cast<float>(max_data[blockIdx.x]));
// #endif
//   if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
// #ifdef __HIPCC__
//   // KERNEL_PRINT("max_data[%lu]=%f", hipBlockIdx_x, static_cast<float>(max_data[blockIdx.x]));
// #else
//   // KERNEL_PRINT("max_data[%d]=%f", blockIdx.x, static_cast<float>(max_data[blockIdx.x]));
// #endif
//   // KERNEL_PRINT("softmax[%d]=%f", blockIdx.x * blockDim.x + threadIdx.x, static_cast<float>(softmax[blockIdx.x * blockDim.x + threadIdx.x]));
//   // // KERNEL_PRINT("max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));
// }

// Make sure that BlockDim <= axis_dim
template <typename T, int BlockDim>
static __global__ void RowReductionForSoftmaxAndCrossEntropy(
    const T* logits_data, const T* labels_data, T* loss_data, T* softmax, int d,
    int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax, labels data view as [n, axis_dim, remain]
  // loss_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  // log_diff_max_sum shares memory with loss
  auto block_log_diff_max_sum = loss_data[blockIdx.x];
  auto tmp = softmax[beg_idx] - block_log_diff_max_sum;
  softmax[beg_idx] = exp_on_device(tmp);
  auto loss = -labels_data[beg_idx] * tmp;
  int step = BlockDim * remain;
  beg_idx += step;
  while (beg_idx < end_idx) {
    tmp = softmax[beg_idx] - block_log_diff_max_sum;
    softmax[beg_idx] = exp_on_device(tmp);
    loss -= (labels_data[beg_idx] * tmp);
    beg_idx += step;
  }

#ifdef __HIPCC__
  loss = BlockReduce<T, BlockDim>(temp_storage).Reduce(loss, hipcub::Sum());
#else
  loss = BlockReduce<T, BlockDim>(temp_storage).Reduce(loss, cub::Sum());
#endif
  if (threadIdx.x == 0) loss_data[blockIdx.x] = loss;
}

template <typename T>
struct HardLabelSoftmaxWithCrossEntropyFunctor {
 public:
  HardLabelSoftmaxWithCrossEntropyFunctor(const int64_t* labels, T* loss,
                                          T* log_softmax, int d, int axis_dim)
      : labels_(labels),
        loss_(loss),
        log_softmax_(log_softmax),
        d_(d),
        axis_dim_(axis_dim) {}

  __device__ void operator()(int idx) const {
    // logits view as [n, axis_dim, remain], where d = axis_dim * remain
    int remain = d_ / axis_dim_;
    int idx_n = idx / d_;
    int idx_axis = (idx % d_) / remain;
    int idx_remain = idx % remain;
    // labels, loss view as [n, remain]
    int idx_lbl = idx_n * remain + idx_remain;

    // KERNEL_PRINT("idx=%d remain=%d idx_n=%d idx_axis=%d idx_remain=%d idx_lbl=%d labels_[idx_lbl]=%lu", 
                  // idx, remain, idx_n, idx_axis, idx_remain, idx_lbl, labels_[idx_lbl]);

    // It also would ignore labels not in range(class_num).
    if (idx_axis != labels_[idx_lbl]) {
      // KERNEL_PRINT("==0== Before idx=%d log_softmax_[idx]=%f", idx, static_cast<float>(log_softmax_[idx]))
      log_softmax_[idx] = exp_on_device(log_softmax_[idx]);
      // KERNEL_PRINT("==0== After idx=%d log_softmax_[idx]=%f", idx, static_cast<float>(log_softmax_[idx]))
    } else {
      // KERNEL_PRINT("==1== Before idx=%d log_softmax_[idx]=%f", idx, static_cast<float>(log_softmax_[idx]))
      auto softmax = log_softmax_[idx];
      log_softmax_[idx] = exp_on_device(softmax);
      loss_[idx_lbl] = -softmax;
      // KERNEL_PRINT("==1== After idx=%d softmax=%f log_softmax_[idx]=%f idx_lbl=%d, loss_[idx_lbl]=%f", 
                    // idx, static_cast<float>(softmax), 
                    // static_cast<float>(log_softmax_[idx]),
                    // idx_lbl,
                    // static_cast<float>(loss_[idx_lbl]))
    }
  }

 private:
  const int64_t* labels_;
  T* loss_;
  T* log_softmax_;
  int d_;
  int axis_dim_;
};

template <typename T>
struct HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx {
 public:
  HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx(const int64_t* labels,
                                                       T* loss, T* log_softmax,
                                                       int d, int axis_dim,
                                                       int ignore_idx)
      : labels_(labels),
        loss_(loss),
        log_softmax_(log_softmax),
        d_(d),
        axis_dim_(axis_dim),
        ignore_idx_(ignore_idx) {}

  __device__ void operator()(int idx) const {
    // logits view as [n, axis_dim, remain], where d = axis_dim * remain
    int remain = d_ / axis_dim_;
    int idx_n = idx / d_;
    int idx_axis = (idx % d_) / remain;
    int idx_remain = idx % remain;
    // labels, loss view as [n, remain]
    int idx_lbl = idx_n * remain + idx_remain;
    if (idx_axis != labels_[idx_lbl] || idx_axis == ignore_idx_) {
      log_softmax_[idx] = exp_on_device(log_softmax_[idx]);
    } else {
      auto softmax = log_softmax_[idx];
      log_softmax_[idx] = exp_on_device(softmax);
      loss_[idx_lbl] = -softmax;
    }
  }

 private:
  const int64_t* labels_;
  T* loss_;
  T* log_softmax_;
  int d_;
  int axis_dim_;
  int ignore_idx_;
};

template <typename T>
static void HardLabelSoftmaxWithCrossEntropy(
    const platform::CUDADeviceContext& ctx, const T* logits_data,
    const int64_t* labels_data, T* loss_data, T* softmax_data, int n, int d,
    int axis_dim, int ignore_idx) {
  constexpr int kMaxBlockDim = 512;
  int block_dim = axis_dim >= kMaxBlockDim
                      ? kMaxBlockDim
                      : (1 << static_cast<int>(std::log2(axis_dim)));
  int grid_dim = n * d / axis_dim;
  auto stream = ctx.stream();
  
  // LOG(INFO) << "===== axis_dim is " << axis_dim; // 5
  // LOG(INFO) << "===== block_dim is " << block_dim; // 4
  // LOG(INFO) << "===== grid_dim is " << grid_dim; // 3
  // LOG(INFO) << "===== ignore_idx is " << ignore_idx; // -1

#ifdef __HIPCC__
#define CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)  \
  case BlockDim: {                                                         \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForMax<T, BlockDim>),   \
        dim3(grid_dim), dim3(BlockDim), 0, stream,                         \
        logits_data, loss_data, d, axis_dim);                              \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForSum<T,  BlockDim>), \
        dim3(grid_dim), dim3(BlockDim), 0, stream,                         \
        logits_data, loss_data, softmax_data, d, axis_dim);                \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForDiff<T,  BlockDim>), \
        dim3(grid_dim), dim3(BlockDim), 0, stream,                         \
        logits_data, loss_data, softmax_data, d, axis_dim);                \
    platform::ForRange<platform::CUDADeviceContext> for_range(ctx, n* d);  \
    if (ignore_idx >= 0 && ignore_idx < axis_dim) {                        \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx<T>(   \
          labels_data, loss_data, softmax_data, d, axis_dim, ignore_idx)); \
    } else {                                                               \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctor<T>(                \
          labels_data, loss_data, softmax_data, d, axis_dim));             \
    }                                                                      \
  } break
#else
#define CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)  \
  case BlockDim: {                                                         \
    RowReductionForMax<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(    \
        logits_data, loss_data, d, axis_dim);                              \
    RowReductionForSum<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(    \
        logits_data, loss_data, softmax_data, d, axis_dim);                \
    RowReductionForDiff<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(   \
        logits_data, loss_data, softmax_data, d, axis_dim);                \
    platform::ForRange<platform::CUDADeviceContext> for_range(ctx, n* d);  \
    if (ignore_idx >= 0 && ignore_idx < axis_dim) {                        \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx<T>(   \
          labels_data, loss_data, softmax_data, d, axis_dim, ignore_idx)); \
    } else {                                                               \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctor<T>(                \
          labels_data, loss_data, softmax_data, d, axis_dim));             \
    }                                                                      \
  } break
#endif

  switch (block_dim) {
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(512);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(256);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(128);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(64);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(32);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(16);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(8);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(4);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(2);
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Block Dimension must be 2^n in softmax_with_cross_entropy_op."));
      break;
  }
#undef CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL
}

template <typename T>
static void SoftmaxWithCrossEntropyFusedKernel(const T* logits_data,
                                               const T* labels_data,
                                               T* softmax_data, T* loss_data,
                                               int n, int d, int axis_dim,
                                               gpuStream_t stream) {
  constexpr int kMaxBlockDim = 512;
  int block_dim = axis_dim >= kMaxBlockDim
                      ? kMaxBlockDim
                      : (1 << static_cast<int>(std::log2(axis_dim)));
  int grid_dim = n * d / axis_dim;

#ifdef __HIPCC__
#define CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)                 \
  case BlockDim:                                                               \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForMax<T, BlockDim>),       \
        dim3(grid_dim), dim3(BlockDim), 0, stream,                             \
        logits_data, loss_data, d, axis_dim);                                  \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForSum<T, BlockDim>),       \
        dim3(grid_dim), dim3(BlockDim), 0, stream,                             \
        logits_data, loss_data, softmax_data, d, axis_dim);                    \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForSoftmaxAndCrossEntropy<  \
        T, BlockDim>), dim3(grid_dim), dim3(BlockDim), 0, stream,              \
        logits_data, labels_data, loss_data, softmax_data, d, axis_dim);       \
    break
#else
#define CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)                 \
  case BlockDim:                                                               \
    RowReductionForMax<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(        \
        logits_data, loss_data, d, axis_dim);                                  \
    RowReductionForSum<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(        \
        logits_data, loss_data, softmax_data, d, axis_dim);                    \
    RowReductionForSoftmaxAndCrossEntropy<                                     \
        T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(                       \
        logits_data, labels_data, loss_data, softmax_data, d, axis_dim);       \
    break
#endif

  switch (block_dim) {
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(512);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(256);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(128);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(64);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(32);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(16);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(8);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(4);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(2);
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Block Dimension must be 2^n in softmax_with_cross_entropy_op."));
      break;
  }

#undef CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL
}

template <typename T>
static void print_data_2d(const T * data, const int64_t numel, const std::vector<int64_t> dims, const std::string name) {
  printf("------------%s------------\n", name.c_str());
  size_t stride = dims[1];
  size_t index = 0;
  while(index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", data[index]);
    } else {
      printf("%d ", data[index]);
    }
    if((index+1) % stride == 0) printf("\n");
    index ++;
  }
}

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");

    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logits->dims()[axis];
    // LOG(INFO) << "axis_dim = " << axis_dim;

    const int n = SizeToAxis(axis, logits->dims());
    const int d = SizeFromAxis(axis, logits->dims());
    // LOG(INFO) << "n = " << n;
    // LOG(INFO) << "d = " << d;
    // LOG(INFO) << "axis = " << axis;

    auto* softmax_data = softmax->mutable_data<T>(context.GetPlace());
    auto* loss_data = loss->mutable_data<T>(context.GetPlace());

    if (axis_dim == 1) {
      math::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(context.cuda_device_context(), softmax, static_cast<T>(1));
      set_constant(context.cuda_device_context(), loss, static_cast<T>(0));
      return;
    }

    auto soft_label = context.Attr<bool>("soft_label");
    auto ignore_index = context.Attr<int>("ignore_index");

    // LOG(INFO) << "soft_label = " << soft_label;
    // LOG(INFO) << "ignore_index = " << ignore_index;

    if (soft_label) {
      auto* logits_data = logits->data<T>();
      auto* labels_data = labels->data<T>();
      SoftmaxWithCrossEntropyFusedKernel(
          logits_data, labels_data, softmax_data, loss_data, n, d, axis_dim,
          context.cuda_device_context().stream());
    } else {
      if (!context.Attr<bool>("numeric_stable_mode")) {
        // CUDNN kernel only suppoer 2-D tensor and perfome softmax on last dim
        Tensor logits_2d, softmax_2d, labels_2d, loss_2d;
        logits_2d.ShareDataWith(*logits).Resize({n, d});
        softmax_2d.ShareDataWith(*softmax).Resize({n, d});
        labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
        loss_2d.ShareDataWith(*loss).Resize({n, 1});
        math::SoftmaxCUDNNFunctor<T>()(context.cuda_device_context(),
                                       &logits_2d, &softmax_2d);
        math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
            context.cuda_device_context(), &loss_2d, &softmax_2d, &labels_2d,
            false, ignore_index, axis_dim);
      } else {
        auto* logits_data = logits->data<T>();
        auto* labels_data = labels->data<int64_t>();
        HardLabelSoftmaxWithCrossEntropy<T>(
            context.cuda_device_context(), logits_data, labels_data, loss_data,
            softmax_data, n, d, axis_dim, ignore_index);
        // for debug
        // size_t logit_numel = static_cast<size_t>(framework::product(logits->dims()));
        // size_t label_numel = static_cast<size_t>(framework::product(labels->dims()));
        // T * logits_cpu = new T[logit_numel];
        // int64_t * label_cpu = new int64_t[label_numel];
        // T * softmax_cpu = new T[logit_numel];
        // T * loss_cpu = new T[label_numel];
        // hipMemcpy(logits_cpu, logits_data, logit_numel * sizeof(T), hipMemcpyDeviceToHost);
        // hipMemcpy(label_cpu, labels_data, label_numel * sizeof(int64_t), hipMemcpyDeviceToHost);
        // hipMemcpy(softmax_cpu, softmax_data, logit_numel * sizeof(T), hipMemcpyDeviceToHost);
        // hipMemcpy(loss_cpu, loss_data, label_numel * sizeof(T), hipMemcpyDeviceToHost);

        // print_data_2d<T>(logits_cpu, logit_numel, framework::vectorize(logits->dims()), "logit");
        // print_data_2d<int64_t>(label_cpu, label_numel, framework::vectorize(labels->dims()), "label");
        // print_data_2d<T>(softmax_cpu, logit_numel, framework::vectorize(logits->dims()), "softmax");
        // print_data_2d<T>(loss_cpu, label_numel, framework::vectorize(labels->dims()), "loss");
      }
    }
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const Tensor* labels = context.Input<Tensor>("Label");
    const T* loss_grad_data =
        context.Input<Tensor>(framework::GradVarName("Loss"))->data<T>();
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* softmax = context.Input<Tensor>("Softmax");
    if (logit_grad != softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }
    T* logit_grad_data = logit_grad->data<T>();

    const int rank = logit_grad->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logit_grad->dims()[axis];

    const int n = SizeToAxis(axis, logit_grad->dims());
    const int d = SizeFromAxis(axis, logit_grad->dims());
    const int remain = d / axis_dim;

    int block = 512;
    auto stream = context.cuda_device_context().stream();
    auto ignore_index = context.Attr<int>("ignore_index");
    if (context.Attr<bool>("soft_label")) {
      int grid = (n * d + block - 1) / block;
      const T* label_data = labels->data<T>();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SoftCrossEntropyGradientKernel<T>), dim3(grid), dim3(block), 0, stream,
          logit_grad_data, loss_grad_data, label_data, n, d, remain);
    } else {
      int grid = (n * remain + block - 1) / block;
      const int64_t* label_data = labels->data<int64_t>();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(CrossEntropyGrad<T>), dim3(grid), dim3(block), 0, stream,
          logit_grad_data, label_data, n, d, remain, ignore_index);
      int num = n * d;
      grid = (num + block - 1) / block;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(Scale<T>), dim3(grid), dim3(block), 0, stream, 
          logit_grad_data, loss_grad_data, num, d, remain, label_data, ignore_index);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(softmax_with_cross_entropy,
    ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
#ifndef PADDLE_WITH_HIP
    ops::SoftmaxWithCrossEntropyCUDAKernel<double>,
#endif
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
#ifndef PADDLE_WITH_HIP
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<double>,
#endif
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>);
