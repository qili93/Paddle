// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA
typedef cudnnDataType_t gpuDnnDataType_t;
typedef cudnnActivationStruct gpuDnnActivationType_t;
typedef cudnnTensorStruct gpuDnnTensorDescType_t;
typedef cudnnFilterStruct gpuDnnFilterDescType_t;
typedef cudnnConvolutionStruct gpuDnnConvolutionDescType_t;
typedef cudnnActivationMode_t gpuDnnActivationMode_t;
#endif

#ifdef PADDLE_WITH_HIP
typedef hipdnnDataType_t gpuDnnDataType_t;
typedef hipdnnActivationDescriptor_t gpuDnnActivationType_t;
typedef hipdnnTensorDescriptor_t gpuDnnTensorDescType_t;
typedef hipdnnTensorDescriptor_t gpuDnnFilterDescType_t;
typedef hipdnnConvolutionDescriptor_t gpuDnnConvolutionDescType_t;
typedef hipdnnActivationMode_t gpuDnnActivationMode_t;
#endif

using framework::Tensor;

template <typename T>
inline gpuDnnDataType_t ToCudnnDataType(const T& t) {
  auto type = framework::ToDataType(t);
  return ToCudnnDataType(type);
}

inline std::vector<int> TransformDimOrder(const std::vector<int>& dims) {
  std::vector<int> transformed_dims(dims.begin(), dims.end());
  int H, W, D, C;
  if (dims.size() == 4) {
    H = dims[1];
    W = dims[2];
    C = dims[3];
    transformed_dims[1] = C;
    transformed_dims[2] = H;
    transformed_dims[3] = W;
  } else {
    D = dims[1];
    H = dims[2];
    W = dims[3];
    C = dims[4];
    transformed_dims[1] = C;
    transformed_dims[2] = D;
    transformed_dims[3] = H;
    transformed_dims[4] = W;
  }
  return transformed_dims;
}

#ifdef PADDLE_WITH_HIP
template <>
inline gpuDnnDataType_t ToCudnnDataType(
    const framework::proto::VarType::Type& t) {
  gpuDnnDataType_t type = HIPDNN_DATA_FLOAT;
  switch (t) {
    case framework::proto::VarType::FP16:
      type = HIPDNN_DATA_HALF;
      break;
    case framework::proto::VarType::FP32:
      type = HIPDNN_DATA_FLOAT;
      break;
    default:
      break;
  }
  return type;
}
#else
template <>
inline gpuDnnDataType_t ToCudnnDataType(
    const framework::proto::VarType::Type& t) {
  gpuDnnDataType_t type = CUDNN_DATA_FLOAT;
  switch (t) {
    case framework::proto::VarType::FP16:
      type = CUDNN_DATA_HALF;
      break;
    case framework::proto::VarType::FP32:
      type = CUDNN_DATA_FLOAT;
      break;
    case framework::proto::VarType::FP64:
      type = CUDNN_DATA_DOUBLE;
      break;
    default:
      break;
  }
  return type;
}
#endif

class ActivationDescriptor {
 public:
  using T = gpuDnnActivationType_t;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_CUDA_SUCCESS(
            dynload::hipdnnDestroyActivationDescriptor(t));
#else
        PADDLE_ENFORCE_CUDA_SUCCESS(
            dynload::cudnnDestroyActivationDescriptor(t));
#endif
        t = nullptr;
      }
    }
  };
  ActivationDescriptor() {
    T* raw_ptr;
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::hipdnnCreateActivationDescriptor(raw_ptr));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateActivationDescriptor(&raw_ptr));
#endif
    desc_.reset(raw_ptr);
  }
  template <typename T>
  void set(gpuDnnActivationMode_t mode, const T& coef) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetActivationDescriptor(
        desc_.get(), mode, HIPDNN_NOT_PROPAGATE_NAN, static_cast<double>(coef), 0.0, 0.0));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetActivationDescriptor(
        desc_.get(), mode, CUDNN_NOT_PROPAGATE_NAN, static_cast<double>(coef)));
#endif
  }

  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class TensorDescriptor {
 public:
  using T = gpuDnnTensorDescType_t;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnDestroyTensorDescriptor(t));
#else
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyTensorDescriptor(t));
#endif
        t = nullptr;
      }
    }
  };
  TensorDescriptor() {
    T* raw_ptr;
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnCreateTensorDescriptor(raw_ptr));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateTensorDescriptor(&raw_ptr));
#endif
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }
  void set(const Tensor& tensor, const int groups = 1) {
    auto dims = framework::vectorize<int>(tensor.dims());
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetTensorNdDescriptor(
        desc_.get(), ToCudnnDataType(tensor.type()), dims_with_group.size(),
        dims_with_group.data(), strides.data()));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        desc_.get(), ToCudnnDataType(tensor.type()), dims_with_group.size(),
        dims_with_group.data(), strides.data()));
#endif
  }

#ifdef PADDLE_WITH_CUDA
// HIP not support cudnnSetTensorNdDescriptorEx
  void set(const Tensor& tensor, const gpuDnnTensorFormat_t format) {
    auto dims = framework::vectorize<int>(tensor.dims());
    std::vector<int> transformed_dims;
    if (format == CUDNN_TENSOR_NHWC) {
      transformed_dims = TransformDimOrder(dims);
    } else {
      transformed_dims = dims;
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
        desc_.get(), format, ToCudnnDataType(tensor.type()),
        transformed_dims.size(), transformed_dims.data()));
  }
#endif

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class FilterDescriptor {
 public:
  using T = gpuDnnFilterDescType_t;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnDestroyFilterDescriptor(t));
#else
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyFilterDescriptor(t));
#endif
        t = nullptr;
      }
    }
  };
  FilterDescriptor() {
    T* raw_ptr;
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnCreateFilterDescriptor(raw_ptr));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateFilterDescriptor(&raw_ptr));
#endif
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

  void set(const Tensor& tensor, const gpuDnnTensorFormat_t format,
           const int groups = 1) {
    auto dims = framework::vectorize<int>(tensor.dims());
    std::vector<int> transformed_dims;
  #ifdef PADDLE_WITH_HIP
    if (format == HIPDNN_TENSOR_NHWC) {
  #else
    if (format == CUDNN_TENSOR_NHWC) {
  #endif
      transformed_dims = TransformDimOrder(dims);
    } else {
      transformed_dims = dims;
    }
    if (groups > 1) {
      transformed_dims[1] = transformed_dims[1] / groups;
    }
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetFilterNdDescriptor(
        desc_.get(), ToCudnnDataType(tensor.type()), format,
        transformed_dims.size(), transformed_dims.data()));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetFilterNdDescriptor(
        desc_.get(), ToCudnnDataType(tensor.type()), format,
        transformed_dims.size(), transformed_dims.data()));
#endif
  }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class ConvolutionDescriptor {
 public:
  using T = gpuDnnConvolutionDescType_t;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_CUDA_SUCCESS(
            dynload::hipdnnDestroyConvolutionDescriptor(t));
#else
        PADDLE_ENFORCE_CUDA_SUCCESS(
            dynload::cudnnDestroyConvolutionDescriptor(t));
#endif
        t = nullptr;
      }
    }
  };
  ConvolutionDescriptor() {
    T* raw_ptr;
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::hipdnnCreateConvolutionDescriptor(raw_ptr));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateConvolutionDescriptor(&raw_ptr));
#endif
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

#ifdef PADDLE_WITH_HIP
  void set(gpuDnnDataType_t dtype, const std::vector<int>& pads,
           const std::vector<int>& strides, const std::vector<int>& dilations,
           const int groups = 1) {
    T* desc = desc_.get();
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::hipdnnSetConvolutionNdDescriptor(
        desc, pads.size(), pads.data(), strides.data(), dilations.data(),
        HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::hipdnnSetConvolutionGroupCount(desc, groups));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipdnnSetConvolutionMathType(
        desc, HIPDNN_DEFAULT_MATH));
    if (dtype == HIPDNN_DATA_HALF) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::hipdnnSetConvolutionMathType(desc,
                                                          HIPDNN_TENSOR_OP_MATH));
    }
  }
#else
  void set(gpuDnnDataType_t dtype, const std::vector<int>& pads,
           const std::vector<int>& strides, const std::vector<int>& dilations,
           const int groups = 1) {
    gpuDnnDataType_t compute_type =
        (dtype == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    T* desc = desc_.get();
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetConvolutionNdDescriptor(
        desc, pads.size(), pads.data(), strides.data(), dilations.data(),
        CUDNN_CROSS_CORRELATION, compute_type));
#if CUDNN_VERSION_MIN(7, 0, 1)
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnSetConvolutionGroupCount(desc, groups));
#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
        desc, CUDNN_DEFAULT_MATH));
    if (dtype == CUDNN_DATA_HALF) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(desc,
                                                         CUDNN_TENSOR_OP_MATH));
    }
#endif
#endif
  }
#endif

 private:
  std::unique_ptr<T, Deleter> desc_;
};

}  // namespace platform
}  // namespace paddle
