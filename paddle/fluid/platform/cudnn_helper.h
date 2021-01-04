/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/macros.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cudnn.h"
typedef cudnnStatus_t gpuDnnStatus_t;
typedef cudnnHandle_t gpuDnnHandle_t;
typedef cudnnDataType_t gpuDnnDataType_t;
typedef cudnnRNNMode_t gpuDnnRNNMode_t;
typedef cudnnTensorDescriptor_t gpuDnnTensorDesc_t;
typedef cudnnRNNDescriptor_t gpuDnnRNNDesc_t;
typedef cudnnConvolutionDescriptor_t gpuDnnConvolutionDesc_t;
typedef cudnnDropoutDescriptor_t gpuDnnDropoutDescriptor_t;
typedef cudnnPoolingDescriptor_t gpuDnnPoolingDesc_t;
typedef cudnnActivationDescriptor_t gpuDnnActivationDesc_t;
typedef cudnnActivationMode_t gpuDnnActivationMode_t;
typedef cudnnCTCLossDescriptor_t gpuDnnCTCLossDesc_t;
typedef cudnnTensorFormat_t gpuDnnTensorFormat_t;
typedef cudnnFilterDescriptor_t gpuDnnFilterDesc_t;
typedef cudnnBatchNormMode_t gpuDnnBatchNormMode_t;
typedef cudnnConvolutionFwdAlgoPerf_t gpuDnnConvolutionFwdAlgoPerf_t;
typedef cudnnConvolutionBwdDataAlgoPerf_t gpuDnnConvolutionBwdDataAlgoPerf_t;
typedef cudnnConvolutionBwdFilterAlgoPerf_t gpuDnnConvolutionBwdFilterAlgoPerf_t;
typedef cudnnConvolutionFwdAlgo_t gpuDnnConvolutionFwdAlgo_t;
typedef cudnnConvolutionBwdDataAlgo_t gpuDnnConvolutionBwdDataAlgo_t;
typedef cudnnConvolutionBwdFilterAlgo_t gpuDnnConvolutionBwdFilterAlgo_t;
#define GPUDNN_DATA_HALF CUDNN_DATA_HALF
#define GPUDNN_DATA_FLOAT CUDNN_DATA_FLOAT
#define GPUDNN_ACTIVATION_RELU CUDNN_ACTIVATION_RELU
#define GPUDNN_ACTIVATION_CLIPPED_RELU CUDNN_ACTIVATION_CLIPPED_RELU
#define GPUDNN_ACTIVATION_SIGMOID CUDNN_ACTIVATION_SIGMOID
#define GPUDNN_ACTIVATION_TANH CUDNN_ACTIVATION_TANH
#define GPUDNN_SOFTMAX_MODE_INSTANCE CUDNN_SOFTMAX_MODE_INSTANCE
#define GPUDNN_SOFTMAX_MODE_CHANNEL CUDNN_SOFTMAX_MODE_CHANNEL
#define GPUDNN_BN_MIN_EPSILON CUDNN_BN_MIN_EPSILON
#define GPUDNN_BATCHNORM_SPATIAL CUDNN_BATCHNORM_SPATIAL
#define GPUDNN_DEFAULT_MATH CUDNN_DEFAULT_MATH
#define GPUDNN_TENSOR_OP_MATH CUDNN_TENSOR_OP_MATH
#define GPUDNN_STATUS_SUCCESS CUDNN_STATUS_SUCCESS
#define GPUDNN_LINEAR_INPUT CUDNN_LINEAR_INPUT
#define GPUDNN_SKIP_INPUT CUDNN_SKIP_INPUT
#define GPUDNN_BIDIRECTIONAL CUDNN_BIDIRECTIONAL
#define GPUDNN_UNIDIRECTIONAL CUDNN_UNIDIRECTIONAL
#define GPUDNN_RNN_ALGO_STANDARD CUDNN_RNN_ALGO_STANDARD
#define GPUDNN_LSTM CUDNN_LSTM
#define GPUDNN_GRU CUDNN_GRU
#define GPUDNN_RNN_RELU CUDNN_RNN_RELU
#define GPUDNN_RNN_TANH CUDNN_RNN_TANH
#endif

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/dynload/hipdnn.h"
typedef hipdnnStatus_t gpuDnnStatus_t;
typedef hipdnnHandle_t gpuDnnHandle_t;
typedef hipdnnDataType_t gpuDnnDataType_t;
typedef hipdnnRNNMode_t gpuDnnRNNMode_t;
typedef hipdnnTensorDescriptor_t gpuDnnTensorDesc_t;
typedef hipdnnRNNDescriptor_t gpuDnnRNNDesc_t;
typedef hipdnnConvolutionDescriptor_t	 gpuDnnConvolutionDesc_t;
typedef hipdnnDropoutDescriptor_t gpuDnnDropoutDescriptor_t;
typedef hipdnnPoolingDescriptor_t gpuDnnPoolingDesc_t;
typedef hipdnnActivationDescriptor_t gpuDnnActivationDesc_t;
typedef hipdnnActivationMode_t gpuDnnActivationMode_t;
typedef hipdnnCTCLossDescriptor_t gpuDnnCTCLossDesc_t;
typedef hipdnnTensorFormat_t gpuDnnTensorFormat_t;
typedef hipdnnFilterDescriptor_t gpuDnnFilterDesc_t;
typedef hipdnnBatchNormMode_t gpuDnnBatchNormMode_t;
typedef hipdnnConvolutionFwdAlgoPerf_t gpuDnnConvolutionFwdAlgoPerf_t;
typedef hipdnnConvolutionBwdDataAlgoPerf_t gpuDnnConvolutionBwdDataAlgoPerf_t;
typedef hipdnnConvolutionBwdFilterAlgoPerf_t gpuDnnConvolutionBwdFilterAlgoPerf_t;
typedef hipdnnConvolutionFwdAlgo_t gpuDnnConvolutionFwdAlgo_t;
typedef hipdnnConvolutionBwdDataAlgo_t gpuDnnConvolutionBwdDataAlgo_t;
typedef hipdnnConvolutionBwdFilterAlgo_t gpuDnnConvolutionBwdFilterAlgo_t;
#define GPUDNN_DATA_HALF HIPDNN_DATA_HALF
#define GPUDNN_DATA_FLOAT HIPDNN_DATA_FLOAT
#define GPUDNN_ACTIVATION_RELU HIPDNN_ACTIVATION_RELU
#define GPUDNN_ACTIVATION_CLIPPED_RELU HIPDNN_ACTIVATION_CLIPPED_RELU
#define GPUDNN_ACTIVATION_SIGMOID HIPDNN_ACTIVATION_SIGMOID
#define GPUDNN_ACTIVATION_TANH HIPDNN_ACTIVATION_TANH
#define GPUDNN_SOFTMAX_MODE_INSTANCE HIPDNN_SOFTMAX_MODE_INSTANCE
#define GPUDNN_SOFTMAX_MODE_CHANNEL HIPDNN_SOFTMAX_MODE_CHANNEL
#define GPUDNN_BN_MIN_EPSILON HIPDNN_BN_MIN_EPSILON
#define GPUDNN_BATCHNORM_SPATIAL HIPDNN_BATCHNORM_SPATIAL
#define GPUDNN_DEFAULT_MATH HIPDNN_DEFAULT_MATH
#define GPUDNN_TENSOR_OP_MATH HIPDNN_TENSOR_OP_MATH
#define GPUDNN_STATUS_SUCCESS HIPDNN_STATUS_SUCCESS
#define GPUDNN_LINEAR_INPUT HIPDNN_LINEAR_INPUT
#define GPUDNN_SKIP_INPUT HIPDNN_SKIP_INPUT
#define GPUDNN_BIDIRECTIONAL HIPDNN_BIDIRECTIONAL
#define GPUDNN_UNIDIRECTIONAL HIPDNN_UNIDIRECTIONAL
#define GPUDNN_RNN_ALGO_STANDARD HIPDNN_RNN_ALGO_STANDARD
#define GPUDNN_LSTM HIPDNN_LSTM
#define GPUDNN_GRU HIPDNN_GRU
#define GPUDNN_RNN_RELU HIPDNN_RNN_RELU
#define GPUDNN_RNN_TANH HIPDNN_RNN_TANH
#endif

namespace paddle {
namespace platform {
struct float16;
}  // namespace platform
}  // namespace paddle

DECLARE_bool(cudnn_deterministic);

namespace paddle {
namespace platform {

inline const char* cudnnGetErrorString(gpuDnnStatus_t status) {
#ifdef PADDLE_WITH_HIP
  switch (status) {
    case HIPDNN_STATUS_SUCCESS:
      return "HIPDNN_STATUS_SUCCESS";
    case HIPDNN_STATUS_NOT_INITIALIZED:
      return "HIPDNN_STATUS_NOT_INITIALIZED";
    case HIPDNN_STATUS_ALLOC_FAILED:
      return "HIPDNN_STATUS_ALLOC_FAILED";
    case HIPDNN_STATUS_BAD_PARAM:
      return "HIPDNN_STATUS_BAD_PARAM";
    case HIPDNN_STATUS_INTERNAL_ERROR:
      return "HIPDNN_STATUS_INTERNAL_ERROR";
    case HIPDNN_STATUS_INVALID_VALUE:
      return "HIPDNN_STATUS_INVALID_VALUE";
    case HIPDNN_STATUS_ARCH_MISMATCH:
      return "HIPDNN_STATUS_ARCH_MISMATCH";
    case HIPDNN_STATUS_MAPPING_ERROR:
      return "HIPDNN_STATUS_MAPPING_ERROR";
    case HIPDNN_STATUS_EXECUTION_FAILED:
      return "HIPDNN_STATUS_EXECUTION_FAILED";
    case HIPDNN_STATUS_NOT_SUPPORTED:
      return "HIPDNN_STATUS_NOT_SUPPORTED";
    case HIPDNN_STATUS_LICENSE_ERROR:
      return "HIPDNN_STATUS_LICENSE_ERROR";
    default:
      return "Unknown cudnn error number";
  }
#else
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    default:
      return "Unknown cudnn error number";
  }
#endif
}

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= ((major)*1000 + (minor)*100 + (patch)))

enum class DataLayout {  // Not use
  kNHWC,
  kNCHW,
  kNCDHW,
  kNDHWC,  // add, liyamei
  kNCHW_VECT_C,
};

enum class PoolingMode {
  kMaximum,
  kMaximumDeterministic,
  kAverageExclusive,
  kAverageInclusive,
};

enum class ActivationMode {
  kNone,  // activation identity
  kSigmoid,
  kRelu,
  kRelu6,
  kReluX,
  kTanh,
  kBandPass,
};
#if defined(PADDLE_WITH_HIP)
inline hipdnnPoolingMode_t GetPoolingMode(const PoolingMode& mode) {
  switch (mode) {
    case PoolingMode::kMaximumDeterministic:
      return HIPDNN_POOLING_MAX;
    case PoolingMode::kAverageExclusive:
      return HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    case PoolingMode::kAverageInclusive:
      return HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case PoolingMode::kMaximum:
      return HIPDNN_POOLING_MAX;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("Unexpected HIPDNN pooling mode."));
  }
}
#elif defined(PADDLE_WITH_CUDA) && CUDNN_VERSION < 6000
#pragma message "CUDNN version under 6.0 is supported at best effort."
#pragma message "We strongly encourage you to move to 6.0 and above."
#pragma message "This message is intended to annoy you enough to update."
#pragma message \
    "please see https://docs.nvidia.com/deeplearning/sdk/cudnn-release-notes/"

inline cudnnPoolingMode_t GetPoolingMode(const PoolingMode& mode) {
  switch (mode) {
    case PoolingMode::kMaximumDeterministic:
      return CUDNN_POOLING_MAX;
    case PoolingMode::kAverageExclusive:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    case PoolingMode::kAverageInclusive:
      return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case PoolingMode::kMaximum:
      return CUDNN_POOLING_MAX;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("Unexpected CUDNN pooling mode."));
  }
}
#else
inline cudnnPoolingMode_t GetPoolingMode(const PoolingMode& mode) {
  switch (mode) {
    case PoolingMode::kMaximumDeterministic:
      return CUDNN_POOLING_MAX_DETERMINISTIC;
    case PoolingMode::kAverageExclusive:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    case PoolingMode::kAverageInclusive:
      return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case PoolingMode::kMaximum:
      return CUDNN_POOLING_MAX;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("Unexpected CUDNN pooling mode."));
  }
}
#endif  // CUDNN_VERSION < 6000

inline ActivationMode StringToActivationMode(const std::string& str) {
  if (str == "identity") {
    return ActivationMode::kNone;
  } else if (str == "sigmoid") {
    return ActivationMode::kSigmoid;
  } else if (str == "relu") {
    return ActivationMode::kRelu;
  } else if (str == "relu6") {
    return ActivationMode::kRelu6;
  } else if (str == "relux") {
    return ActivationMode::kReluX;
  } else if (str == "tanh") {
    return ActivationMode::kTanh;
  } else if (str == "bandpass") {
    return ActivationMode::kBandPass;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unknown CUDNN activation string: %s.", str));
  }
}

template <typename T>
class CudnnDataType;

template <>
class CudnnDataType<float16> {
 public:
  static const gpuDnnDataType_t type = GPUDNN_DATA_HALF;
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class CudnnDataType<float> {
 public:
  static const gpuDnnDataType_t type = GPUDNN_DATA_FLOAT;
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

#ifdef PADDLE_WITH_CUDA
// HIPDNN not support DATA_DOUBLE
template <>
class CudnnDataType<double> {
 public:
  static const gpuDnnDataType_t type = CUDNN_DATA_DOUBLE;
  using ScalingParamType = const double;
  using BatchNormParamType = double;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};
#endif

#ifdef PADDLE_WITH_HIP
inline gpuDnnTensorFormat_t GetCudnnTensorFormat(
    const DataLayout& order) {  // Not use
  switch (order) {
    case DataLayout::kNHWC:
      return HIPDNN_TENSOR_NHWC;
    case DataLayout::kNCHW:
      return HIPDNN_TENSOR_NCHW;
    case DataLayout::kNCDHW:
      return HIPDNN_TENSOR_NCHW;  // NOTE: cudnn treat NdTensor as the same
    case DataLayout::kNDHWC:
      return HIPDNN_TENSOR_NHWC;  // add, liyamei
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "HIPDNN has no equivalent dataLayout for input order."));
  }
  return HIPDNN_TENSOR_NCHW;
}
#else
inline gpuDnnTensorFormat_t GetCudnnTensorFormat(
    const DataLayout& order) {  // Not use
  switch (order) {
    case DataLayout::kNHWC:
      return CUDNN_TENSOR_NHWC;
    case DataLayout::kNCHW:
      return CUDNN_TENSOR_NCHW;
    case DataLayout::kNCDHW:
      return CUDNN_TENSOR_NCHW;  // NOTE: cudnn treat NdTensor as the same
    case DataLayout::kNDHWC:
      return CUDNN_TENSOR_NHWC;  // add, liyamei
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "CUDNN has no equivalent dataLayout for input order."));
  }
  return CUDNN_TENSOR_NCHW;
}
#endif

class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnCreateTensorDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateTensorDescriptor(&desc_));
#endif
  }
  ~ScopedTensorDescriptor() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnDestroyTensorDescriptor(desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyTensorDescriptor(desc_));
#endif
  }

  inline gpuDnnTensorDesc_t descriptor(const gpuDnnTensorFormat_t format,
                                       const gpuDnnDataType_t type,
                                       const std::vector<int>& dims,
                                       const int groups = 1) {
    // the format is not used now, will add later
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    // Update tensor descriptor dims setting if groups > 1
    // NOTE: Here, Assume using NCHW or NCDHW order
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }

#ifdef PADDLE_WITH_HIP
    if (dims.size() == 4) {
      if (format == HIPDNN_TENSOR_NCHW) {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetTensorNdDescriptor(
            desc_, type, dims_with_group.size(), dims_with_group.data(),
            strides.data()));
      } else {  // CUDNN_TENSOR_NHWC
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetTensor4dDescriptor(
            desc_, format, type, dims[0], dims[3], dims[1], dims[2]));
      }
    } else if (dims.size() == 5) {
      if (format == HIPDNN_TENSOR_NCHW) {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetTensorNdDescriptor(
            desc_, type, dims_with_group.size(), dims_with_group.data(),
            strides.data()));
      }
    }
#else
    if (dims.size() == 4) {
      if (format == CUDNN_TENSOR_NCHW) {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
            desc_, type, dims_with_group.size(), dims_with_group.data(),
            strides.data()));
      } else {  // CUDNN_TENSOR_NHWC
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensor4dDescriptor(
            desc_, format, type, dims[0], dims[3], dims[1], dims[2]));
      }
    } else if (dims.size() == 5) {
      if (format == CUDNN_TENSOR_NCHW) {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
            desc_, type, dims_with_group.size(), dims_with_group.data(),
            strides.data()));
      } else {  // CUDNN_TENSOR_NHWC
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
            desc_, format, type, dims.size(), dims.data()));
      }
    }
#endif
    return desc_;
  }

  template <typename T>
  inline gpuDnnTensorDesc_t descriptor(const DataLayout& order,
                                       const std::vector<int>& dims,
                                       const int groups = 1) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type, dims,
                      groups);
  }

  inline gpuDnnTensorDesc_t descriptor(const gpuDnnDataType_t cudnn_type,
                                       const std::vector<int>& dim,
                                       const std::vector<int>& stride) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetTensorNdDescriptor(
        desc_, cudnn_type, dim.size(), dim.data(), stride.data()));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        desc_, cudnn_type, dim.size(), dim.data(), stride.data()));
#endif
    return desc_;
  }

  template <typename T>
  inline gpuDnnTensorDesc_t descriptor(const std::vector<int>& dim,
                                       const std::vector<int>& stride) {
    return descriptor(CudnnDataType<T>::type, dim, stride);
  }

  inline gpuDnnTensorDesc_t desc() { return desc_; }

 private:
  gpuDnnTensorDesc_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 7201
class ScopedRNNTensorDescriptor {
 public:
  ScopedRNNTensorDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateRNNDataDescriptor(&desc_));
  }

  ~ScopedRNNTensorDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyRNNDataDescriptor(desc_));
  }

  inline cudnnRNNDataDescriptor_t descriptor(
      const gpuDnnDataType_t cudnn_type, int max_seq_length, int batch_size,
      int input_size, bool time_major, const std::vector<int>& seq_length) {
    static double padding_fill = 0.0f;
    cudnnRNNDataLayout_t layout;

    if (time_major) {
      layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
    } else {
      layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
    }

    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetRNNDataDescriptor(
        desc_, cudnn_type, layout, max_seq_length, batch_size, input_size,
        seq_length.data(), static_cast<void*>(&padding_fill)));

    return desc_;
  }

  template <typename T>
  inline cudnnRNNDataDescriptor_t descriptor(
      int max_length, int batch_size, int input_size, bool time_major,
      const std::vector<int>& seq_length) {
    return descriptor(CudnnDataType<T>::type, max_length, batch_size,
                      input_size, time_major, seq_length);
  }

  inline cudnnRNNDataDescriptor_t desc() { return desc_; }

 private:
  cudnnRNNDataDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedRNNTensorDescriptor);
};
#endif

class ScopedDropoutDescriptor {
 public:
  ScopedDropoutDescriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnCreateDropoutDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateDropoutDescriptor(&desc_));
#endif
  }
  ~ScopedDropoutDescriptor() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnDestroyDropoutDescriptor	(desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyDropoutDescriptor(desc_));
#endif
  }

  inline gpuDnnDropoutDescriptor_t descriptor(const gpudnnHandle_t& handle,
                                             const platform::Place& place,
                                             bool initialized,
                                             float dropout_prob_,
                                             framework::Tensor* dropout_state_,
                                             int seed, size_t state_size) {
    if (dropout_state_ == nullptr) {  // for no dropout or test
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetDropoutDescriptor(
          desc_, handle, 0 /* dropout */, nullptr, 0 /* state_size */,
          0 /* seed */));
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetDropoutDescriptor(
          desc_, handle, 0 /* dropout */, nullptr, 0 /* state_size */,
          0 /* seed */));
#endif
      return desc_;
    }
    auto* dropout_state_data = dropout_state_->data<uint8_t>();
    if (!initialized) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetDropoutDescriptor(
          desc_, handle, dropout_prob_, dropout_state_data, state_size, seed));
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetDropoutDescriptor(
          desc_, handle, dropout_prob_, dropout_state_data, state_size, seed));
#endif
    } else {
      auto dropout_state_dims = dropout_state_->dims();
      state_size = dropout_state_dims[0];
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnRestoreDropoutDescriptor(
          desc_, handle, dropout_prob_, dropout_state_data, state_size, 0));
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnRestoreDropoutDescriptor(
          desc_, handle, dropout_prob_, dropout_state_data, state_size, 0));
#endif
    }
    return desc_;
  }
  inline gpuDnnDropoutDescriptor_t desc() { return desc_; }

 private:
  gpuDnnDropoutDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedDropoutDescriptor);
};

class ScopedRNNDescriptor {
 public:
  ScopedRNNDescriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnCreateRNNDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateRNNDescriptor(&desc_));
#endif
  }
  ~ScopedRNNDescriptor() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnDestroyRNNDescriptor(desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyRNNDescriptor(desc_));
#endif
  }

  inline gpuDnnRNNDesc_t desc() { return desc_; }

 private:
  gpuDnnRNNDesc_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedRNNDescriptor);
};

class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnCreateFilterDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateFilterDescriptor(&desc_));
#endif
  }
  ~ScopedFilterDescriptor() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnDestroyFilterDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyFilterDescriptor(desc_));
#endif
  }

  inline gpuDnnFilterDesc_t descriptor(const gpuDnnTensorFormat_t format,
                                            const gpuDnnDataType_t type,
                                            const std::vector<int>& kernel,
                                            const int groups = 1) {
    // filter layout: MCHW(MCDHW), where M is the number of
    // output image channels, C is the number of input image channels,
    // D is the depth of the filter, H is the height of the filter, and W is the
    // width of the filter.
    std::vector<int> kernel_with_group(kernel.begin(), kernel.end());
    if (groups > 1) {
      kernel_with_group[0] /= groups;
      // NOTE: input filter(C) of the filter is already asserted to be C/groups.
    }
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetFilterNdDescriptor(
        desc_, type, format, kernel_with_group.size(),
        kernel_with_group.data()));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetFilterNdDescriptor(
        desc_, type, format, kernel_with_group.size(),
        kernel_with_group.data()));
#endif
    return desc_;
  }

  template <typename T>
  inline gpuDnnFilterDesc_t descriptor(const DataLayout& order,
                                            const std::vector<int>& kernel,
                                            const int groups = 1) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type,
                      kernel, groups);
  }

  inline gpuDnnFilterDesc_t desc() { return desc_; }

 private:
  gpuDnnFilterDesc_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::hipdnnCreateConvolutionDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateConvolutionDescriptor(&desc_));
#endif
  }
  ~ScopedConvolutionDescriptor() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::hipdnnDestroyConvolutionDescriptor(desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyConvolutionDescriptor(desc_));
#endif
  }

  inline gpuDnnConvolutionDesc_t descriptor(gpuDnnDataType_t type,
                                            const std::vector<int>& pads,
                                            const std::vector<int>& strides,
                                            const std::vector<int>& dilations) {
    PADDLE_ENFORCE_EQ(pads.size(), strides.size(),
                      platform::errors::InvalidArgument(
                          "The size of pads and strides should be equal. But "
                          "received size of pads is %d, size of strides is %d.",
                          pads.size(), strides.size()));
    PADDLE_ENFORCE_EQ(
        pads.size(), dilations.size(),
        platform::errors::InvalidArgument(
            "The size of pads and dilations should be equal. But received size "
            "of pads is %d, size of dilations is %d.",
            pads.size(), dilations.size()));

#if defined(PADDLE_WITH_CUDA) && !CUDNN_VERSION_MIN(6, 0, 0)
    // cudnn v5 does not support dilation conv, the argument is called upscale
    // instead of dilations and it is must be one.
    for (size_t i = 0; i < dilations.size(); ++i) {
      PADDLE_ENFORCE_EQ(dilations[i], 1,
                        platform::errors::InvalidArgument(
                            "Dilations conv is not supported in this cuDNN "
                            "version(%d.%d.%d).",
                            CUDNN_VERSION / 1000, CUDNN_VERSION % 1000 / 100,
                            CUDNN_VERSION % 100));
    }
#endif

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetConvolutionNdDescriptor(
        desc_, pads.size(), pads.data(), strides.data(), dilations.data(),
        HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));
#else
    gpuDnnDataType_t compute_type =
        (type == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : GPUDNN_DATA_FLOAT;
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetConvolutionNdDescriptor(
        desc_, pads.size(), pads.data(), strides.data(), dilations.data(),
        CUDNN_CROSS_CORRELATION, compute_type));
#endif

    return desc_;
  }

  template <typename T>
  inline gpuDnnConvolutionDesc_t descriptor(const std::vector<int>& pads,
                                            const std::vector<int>& strides,
                                            const std::vector<int>& dilations) {
    return descriptor(CudnnDataType<T>::type, pads, strides, dilations);
  }

 private:
  gpuDnnConvolutionDesc_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnCreatePoolingDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreatePoolingDescriptor(&desc_));
#endif
  }
  ~ScopedPoolingDescriptor() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnDestroyPoolingDescriptor(desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyPoolingDescriptor(desc_));
#endif
  }

  inline gpuDnnPoolingDesc_t descriptor(const PoolingMode& mode,
                                        const std::vector<int>& kernel,
                                        const std::vector<int>& pads,
                                        const std::vector<int>& strides) {
    PADDLE_ENFORCE_EQ(kernel.size(), pads.size(),
                      platform::errors::InvalidArgument(
                          "The size of kernel and pads should be equal. But "
                          "received size of kernel is %d, size of pads is %d.",
                          kernel.size(), pads.size()));
    PADDLE_ENFORCE_EQ(
        kernel.size(), strides.size(),
        platform::errors::InvalidArgument(
            "The size of kernel and strides should be equal. But "
            "received size of kernel is %d, size of strides is %d.",
            kernel.size(), strides.size()));
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetPoolingNdDescriptor(
        desc_, (GetPoolingMode(mode)),
        HIPDNN_PROPAGATE_NAN,  // Always propagate nans.
        kernel.size(), kernel.data(), pads.data(), strides.data()));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetPoolingNdDescriptor(
        desc_, (GetPoolingMode(mode)),
        CUDNN_PROPAGATE_NAN,  // Always propagate nans.
        kernel.size(), kernel.data(), pads.data(), strides.data()));
#endif
    return desc_;
  }

 private:
  gpuDnnPoolingDesc_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

#ifdef PADDLE_WITH_CUDA
class ScopedSpatialTransformerDescriptor {
 public:
  ScopedSpatialTransformerDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateSpatialTransformerDescriptor(&desc_));
  }
  ~ScopedSpatialTransformerDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroySpatialTransformerDescriptor(desc_));
  }

  template <typename T>
  inline cudnnSpatialTransformerDescriptor_t descriptor(const int nbDims,
                                                        const int dimA[]) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetSpatialTransformerNdDescriptor(
        desc_, CUDNN_SAMPLER_BILINEAR, CudnnDataType<T>::type, nbDims, dimA));
    return desc_;
  }

 private:
  cudnnSpatialTransformerDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedSpatialTransformerDescriptor);
};
#endif

class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::hipdnnCreateActivationDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateActivationDescriptor(&desc_));
#endif
  }
  ~ScopedActivationDescriptor() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::hipdnnDestroyActivationDescriptor(desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyActivationDescriptor(desc_));
#endif
  }

  template <typename T>
  inline gpuDnnActivationDesc_t descriptor(
      const std::string& act, double value_max = static_cast<double>(0.)) {
    double relu_ceiling = 0.0;
    ActivationMode activation_mode = StringToActivationMode(act);
    gpuDnnActivationMode_t mode;
#ifdef PADDLE_WITH_HIP
    switch (activation_mode) {
      case ActivationMode::kNone:
        mode = HIPDNN_ACTIVATION_PATHTRU;
        break;
      case ActivationMode::kRelu6:
        relu_ceiling = 6.0;
        mode = HIPDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case ActivationMode::kReluX:
        relu_ceiling = value_max;
        mode = HIPDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case ActivationMode::kRelu:
        mode = HIPDNN_ACTIVATION_RELU;
        break;
      case ActivationMode::kSigmoid:
        mode = HIPDNN_ACTIVATION_SIGMOID;
        break;
      case ActivationMode::kTanh:
        mode = HIPDNN_ACTIVATION_TANH;
        break;
      default:
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unrecognized HIPDNN activation mode: %d.",
            static_cast<int>(activation_mode)));
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetActivationDescriptor(
        desc_, mode, HIPDNN_NOT_PROPAGATE_NAN, relu_ceiling, 1, 0));
#else
    switch (activation_mode) {
#if CUDNN_VERSION >= 7100
      case ActivationMode::kNone:
        mode = CUDNN_ACTIVATION_IDENTITY;
        break;
#endif
      case ActivationMode::kRelu6:
        relu_ceiling = 6.0;
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case ActivationMode::kReluX:
        relu_ceiling = value_max;
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case ActivationMode::kRelu:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case ActivationMode::kSigmoid:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case ActivationMode::kTanh:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      default:
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unrecognized CUDNN activation mode: %d.",
            static_cast<int>(activation_mode)));
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetActivationDescriptor(
        desc_, mode, CUDNN_NOT_PROPAGATE_NAN, relu_ceiling));
#endif
    return desc_;
  }

 private:
  gpuDnnActivationDesc_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedActivationDescriptor);
};

inline bool CanCUDNNBeUsed(const framework::ExecutionContext& ctx) {
  bool use_cudnn = ctx.Attr<bool>("use_cudnn");
  use_cudnn &= paddle::platform::is_gpu_place(ctx.GetPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (use_cudnn) {
    auto& dev_ctx = ctx.device_context<platform::CUDADeviceContext>();
    use_cudnn &= dev_ctx.cudnn_handle() != nullptr;
  }
#endif
  return use_cudnn;
}

#if defined(PADDLE_WITH_HIP) || \
    (defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 7001)
class ScopedCTCLossDescriptor {
 public:
  ScopedCTCLossDescriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnCreateCTCLossDescriptor(&desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateCTCLossDescriptor(&desc_));
#endif
  }
  ~ScopedCTCLossDescriptor() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnDestroyCTCLossDescriptor(desc_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyCTCLossDescriptor(desc_));
#endif
  }

  template <typename T>
  inline gpuDnnCTCLossDesc_t descriptor() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::hipdnnSetCTCLossDescriptor(desc_, CudnnDataType<T>::type, 0, false));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnSetCTCLossDescriptor(desc_, CudnnDataType<T>::type));
#endif
    return desc_;
  }

 private:
  gpuDnnCTCLossDesc_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedCTCLossDescriptor);
};
#endif

}  // namespace platform
}  // namespace paddle
