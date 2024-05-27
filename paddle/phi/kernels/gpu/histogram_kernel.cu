// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/histogram_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

// for debug
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

// ----------------------- for debug: start -----------------------
template <typename T>
inline std::string GetTensorDesc(const phi::DenseTensor& tensor) {
  std::string dtype_str = phi::DataTypeToString(tensor.dtype());
  std::stringstream ss;
  if (tensor.initialized()) {
    ss << ": initialized: 1, place: " << tensor.place()
       << ", dtype: " << dtype_str << ", format: " << tensor.layout()
       << ", dims: [" << tensor.dims() << "]"
       << ", capacity: <" << tensor.capacity() << ">";
  } else {
    ss << ", initialized: 0, place: Unknown"
       << ", dtype: " << dtype_str << ", format: " << tensor.layout()
       << ", dims: [" << tensor.dims() << "]";
  }
  return ss.str();
}

template <typename T, typename Context>
void DumpTensor2String(const Context& dev_ctx,
                       const phi::DenseTensor& tensor) {
  auto tensor_desc = GetTensorDesc<T>(tensor);

  std::vector<T> tensor_data;
  if (tensor.place() == phi::CPUPlace()) {
    phi::TensorToVector(tensor, dev_ctx, &tensor_data);
  } else {
    phi::DenseTensor cpu_tensor;
    phi::Copy(dev_ctx, tensor, phi::CPUPlace(), true, &cpu_tensor);
    phi::TensorToVector(cpu_tensor, dev_ctx, &tensor_data);
  }

  LOG(INFO) << "TensorDesc = { " << tensor_desc << " }\n";
  for (int i = 0; i < tensor_data.size(); ++i) {
    LOG(INFO) << "TensorData [" << i << "] = { " << tensor_data[i] << " }\n";
  }
}

#define KPRINT(__FORMAT, ...)              \
  printf("%03d: [tid.x=<%d> tid.y=<%d> bid.x=<%d> bid.y=<%d>]: " __FORMAT "\n", \
  __LINE__, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ##__VA_ARGS__);

// ----------------------- for debug: finish -----------------------

using IndexType = int64_t;
using phi::PADDLE_CUDA_NUM_THREADS;

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T, typename IndexType>
__device__ static IndexType GetBin(T input_value,
                                   T min_value,
                                   T max_value,
                                   int64_t nbins) {
  IndexType bin = static_cast<int64_t>((input_value - min_value) * nbins /
                                   (max_value - min_value));
  KPRINT("input_value=%lld, min_value=%lld, max_value=%lld, nbins=%lld, bin=%lld", input_value, min_value, max_value, nbins, bin);
  IndexType output_index = bin < nbins - 1 ? bin : nbins - 1;
  KPRINT("bin=%d, output_index=%d", bin, output_index);
  return output_index;
}

template <typename T, typename IndexType>
__global__ void KernelHistogram(const T* input,
                                const int total_elements,
                                const int64_t nbins,
                                const T* min_value,
                                const T* max_value,
                                int64_t* output) {
  extern __shared__ int64_t buf_hist[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    buf_hist[i] = 0;
    KPRINT("buf_hist[%d]=%lld", i, buf_hist[i]);
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(input_index, total_elements) {
    // const IndexType input_index = threadIdx.x + blockIdx.x * blockDim.x;
    const auto input_value = input[input_index];
    KPRINT("input_index=%d, input_value=%lld, min_value=%lld, max_value=%lld", input_index, input_value, *min_value, *max_value);
    if (input_value >= *min_value && input_value <= *max_value) {
      const IndexType output_index =
          GetBin<T, IndexType>(input_value, *min_value, *max_value, nbins);
      phi::CudaAtomicAdd(&buf_hist[output_index], 1);
      KPRINT("output_index=%d, buf_hist[output_index]=%lld", output_index, buf_hist[output_index]);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    KPRINT("buf_hist[%d]=%lld", i, buf_hist[i]);
    phi::CudaAtomicAdd(&output[i], buf_hist[i]);
  }
}

template <typename T>
__global__ void KernelMinMax(const T* input,
                             const int numel,
                             const int block_num,
                             T* min_ptr,
                             T* max_ptr) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t i = index;
  T min_value = static_cast<T>(i < numel ? input[i] : input[0]);
  T max_value = static_cast<T>(i < numel ? input[i] : input[0]);

  for (; i < numel; i += blockDim.x * gridDim.x) {
    T value = static_cast<T>(input[i]);
    min_value = value < min_value ? value : min_value;
    max_value = value > max_value ? value : max_value;
  }
  if (max_ptr && min_ptr) {
    __syncthreads();
    T block_min_value = phi::funcs::BlockReduceMin<T>(min_value, FINAL_MASK);
    T block_max_value = phi::funcs::BlockReduceMax<T>(max_value, FINAL_MASK);

    if (threadIdx.x == 0) {
      min_ptr[blockIdx.x] = block_min_value;
      max_ptr[blockIdx.x] = block_max_value;
    }
  }
  __syncthreads();
  if (index == 0) {
    if (min_ptr && max_ptr) {
      min_value = min_ptr[0];
      max_value = max_ptr[0];
      for (int64_t i = 1; i < block_num; i++) {
        min_ptr[0] = min_ptr[i] < min_value ? min_ptr[i] : min_value;
        max_ptr[0] = max_ptr[i] > max_value ? max_ptr[i] : max_value;
      }
      if (min_ptr[0] == max_ptr[0]) {
        min_ptr[0] = min_ptr[0] - 1;
        max_ptr[0] = max_ptr[0] + 1;
      }
    }
  }
}

template <typename T>
__global__ void KernelMinMax(const T min_value,
                             const T max_value,
                             T* min_ptr,
                             T* max_ptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    min_ptr[0] = min_value;
    max_ptr[0] = max_value;
  }
}

template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     int64_t bins,
                     int min,
                     int max,
                     DenseTensor* output) {
  auto& nbins = bins;
  auto& minval = min;
  auto& maxval = max;

  LOG(INFO) << "nbins = " << nbins;
  LOG(INFO) << "minval = " << minval;
  LOG(INFO) << "maxval = " << maxval;

  const T* input_data = input.data<T>();
  const int input_numel = input.numel();

  LOG(INFO) << "input_numel = " << input_numel;

  int64_t* out_data = dev_ctx.template Alloc<int64_t>(output);
  phi::funcs::SetConstant<Context, int64_t>()(
      dev_ctx, output, static_cast<int64_t>(0));

  if (input_data == nullptr) return;

  T output_min = static_cast<T>(minval);
  T output_max = static_cast<T>(maxval);

  DenseTensor min_max;
  int block_num = GET_BLOCKS(input_numel);
  min_max.Resize({2 * block_num});
  LOG(INFO) << "block_num = " << block_num;
  auto* min_block_ptr = dev_ctx.template Alloc<T>(&min_max);
  auto* max_block_ptr = min_block_ptr + block_num;

  // LOG(INFO) << "output_min = " << output_min;
  // LOG(INFO) << "output_max = " << output_max;
  LOG(INFO) << "min_block_ptr = " << min_block_ptr;
  LOG(INFO) << "max_block_ptr = " << max_block_ptr;
  if (min == max) {
    KernelMinMax<T><<<GET_BLOCKS(input_numel),
                      PADDLE_CUDA_NUM_THREADS,
                      0,
                      dev_ctx.stream()>>>(
        input_data, input_numel, block_num, min_block_ptr, max_block_ptr);
  } else {
    KernelMinMax<T><<<1, 1, 0, dev_ctx.stream()>>>(
        output_min, output_max, min_block_ptr, max_block_ptr);
  }

  // copy min max value from GPU to CPU to check
  std::vector<T> min_max_vec;
  phi::TensorToVector(min_max, dev_ctx, &min_max_vec);
  output_min = min_max_vec[0];
  output_max = min_max_vec[1];
  LOG(INFO) << "output_min = " << output_min;
  LOG(INFO) << "output_max = " << output_max;

  // check if out of range
  double range = static_cast<double>(output_max - output_min);
  double range_value = static_cast<double>(output_max) - static_cast<double>(output_min);
  LOG(INFO) << "range = " << range;
  LOG(INFO) << "range_value = " << range_value;
  LOG(INFO) << "std::numeric_limits<T> = " << std::numeric_limits<T>::max();
  PADDLE_ENFORCE_LT(range_value, static_cast<double>(std::numeric_limits<T>::max()),
      phi::errors::InvalidArgument(
            "The range of max - min is out of range for target type, "
            "current kernel type is %s, the range should less than %f "
            "but now min is %f, max is %f.",
            typeid(T).name(),
            std::numeric_limits<T>::max(),
            output_min, output_max));

  LOG(INFO) << "--------- dump: min_max begin ---------";
  DumpTensor2String<T, Context>(dev_ctx, min_max);
  LOG(INFO) << "--------- dump: min_max finish ---------";

  PADDLE_ENFORCE_EQ((std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max)) ||
                     std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max))),
                    false,
                    phi::errors::OutOfRange("range of min, max is not finite"));
  PADDLE_ENFORCE_GE(
      output_max,
      output_min,
      phi::errors::InvalidArgument(
          "max must be larger or equal to min. If min and max are both zero, "
          "the minimum and maximum values of the data are used. "
          "But received max is %d, min is %d",
          maxval,
          minval));

  auto stream = dev_ctx.stream();
  LOG(INFO) << "input_numel = " << input_numel;
  LOG(INFO) << "GET_BLOCKS(input_numel) = " << GET_BLOCKS(input_numel);
  LOG(INFO) << "PADDLE_CUDA_NUM_THREADS = " << PADDLE_CUDA_NUM_THREADS;
  LOG(INFO) << "nbins = " << nbins;
  LOG(INFO) << "nbins * sizeof(int64_t) = " << nbins * sizeof(int64_t);
  KernelHistogram<T, IndexType><<<GET_BLOCKS(input_numel), // gridDim = 1
                                  PADDLE_CUDA_NUM_THREADS, // blockDim = 512
                                  nbins * sizeof(int64_t), // sharedMemSize = 8
                                  stream>>>(
      input_data, input_numel, nbins, min_block_ptr, max_block_ptr, out_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(histogram,
                   GPU,
                   ALL_LAYOUT,
                   phi::HistogramKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::INT64);
}
