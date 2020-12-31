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
#include <glog/logging.h>

#include <hipdnn.h>
#include <mutex>  // NOLINT
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag hipdnn_dso_flag;
extern void* hipdnn_dso_handle;
extern bool HasCUDNN();

extern void EnforceHIPDNNLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP(__name)                            \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {        \
      using hipdnn_func = decltype(&::__name);                              \
      std::call_once(hipdnn_dso_flag, []() {                                \
        hipdnn_dso_handle = paddle::platform::dynload::GetCUDNNDsoHandle(); \
      });                                                                   \
      EnforceHIPDNNLoaded(#__name);                                          \
      static void* p_##__name = dlsym(hipdnn_dso_handle, #__name);          \
      return reinterpret_cast<hipdnn_func>(p_##__name)(args...);            \
    }                                                                       \
  };                                                                        \
  extern struct DynLoad__##__name __name


/**
 * include all needed hipdnn functions in HPPL
 * different hipdnn version has different interfaces
 **/
#define HIPDNN_DNN_ROUTINE_EACH(__macro)                   \
  __macro(hipdnnSetTensor4dDescriptor);                    \
  __macro(hipdnnSetTensor4dDescriptorEx);                  \
  __macro(hipdnnSetTensorNdDescriptor);                    \
  __macro(hipdnnGetTensorNdDescriptor);                    \
  __macro(hipdnnCreateTensorDescriptor);                   \
  __macro(hipdnnDestroyTensorDescriptor);                  \
  __macro(hipdnnCreateFilterDescriptor);                   \
  __macro(hipdnnSetFilter4dDescriptor);                    \
  __macro(hipdnnSetFilterNdDescriptor);                    \
  __macro(hipdnnGetFilterNdDescriptor);                    \
  __macro(hipdnnSetPooling2dDescriptor);                   \
  __macro(hipdnnGetPooling2dDescriptor);                   \
  __macro(hipdnnDestroyFilterDescriptor);                  \
  __macro(hipdnnCreateConvolutionDescriptor);              \
  __macro(hipdnnDestroyConvolutionDescriptor);              \
  __macro(hipdnnCreatePoolingDescriptor);                  \
  __macro(hipdnnDestroyPoolingDescriptor);                 \
  __macro(hipdnnSetPoolingNdDescriptor);                 \
  __macro(hipdnnSetConvolution2dDescriptor);               \
  __macro(hipdnnSetConvolutionNdDescriptor);               \
  __macro(hipdnnGetConvolutionNdDescriptor);               \
  __macro(hipdnnDeriveBNTensorDescriptor);                 \
  __macro(hipdnnCreate);                                   \
  __macro(hipdnnDestroy);                                  \
  __macro(hipdnnSetStream);                                \
  __macro(hipdnnActivationForward);                        \
  __macro(hipdnnActivationBackward);                       \
  __macro(hipdnnConvolutionForward);                       \
  __macro(hipdnnConvolutionBackwardBias);                  \
  __macro(hipdnnGetConvolutionForwardWorkspaceSize);       \
  __macro(hipdnnPoolingForward);                           \
  __macro(hipdnnPoolingBackward);                          \
  __macro(hipdnnSoftmaxBackward);                          \
  __macro(hipdnnSoftmaxForward);                           \
  __macro(hipdnnGetVersion);                               \
  __macro(hipdnnFindConvolutionForwardAlgorithmEx);        \
  __macro(hipdnnFindConvolutionBackwardFilterAlgorithmEx); \
  __macro(hipdnnFindConvolutionBackwardFilterAlgorithm);   \
  __macro(hipdnnFindConvolutionBackwardDataAlgorithmEx);   \
  __macro(hipdnnGetErrorString);                           \
  __macro(hipdnnCreateDropoutDescriptor);                  \
  __macro(hipdnnDestroyDropoutDescriptor);                  \
  __macro(hipdnnRestoreDropoutDescriptor);                 \
  __macro(hipdnnDropoutGetStatesSize);                     \
  __macro(hipdnnSetDropoutDescriptor);                     \
  __macro(hipdnnCreateRNNDescriptor);                      \
  __macro(hipdnnGetRNNParamsSize);                         \
  __macro(hipdnnGetRNNWorkspaceSize);                      \
  __macro(hipdnnGetRNNTrainingReserveSize);                \
  __macro(hipdnnRNNForwardTraining);                       \
  __macro(hipdnnRNNBackwardData);                          \
  __macro(hipdnnRNNBackwardWeights);                       \
  __macro(hipdnnRNNForwardInference);                      \
  __macro(hipdnnDestroyRNNDescriptor);

// HIP platform not supported, refer to the following url:
// https://github.com/ROCm-Developer-Tools/HIP/blob/roc-3.5.x/docs/markdown/CUDNN_API_supported_by_HIP.md
  // __macro(cudnnGetConvolutionNdForwardOutputDim);
  // __macro(cudnnGetPoolingNdDescriptor);
  // __macro(cudnnDestroyConvolutionDescriptor);
  // __macro(cudnnGetConvolutionNdDescriptor);
  // __macro(cudnnCreateSpatialTransformerDescriptor); 
  // __macro(cudnnSetSpatialTransformerNdDescriptor);  
  // __macro(cudnnDestroySpatialTransformerDescriptor);
  // __macro(cudnnSpatialTfGridGeneratorForward);      
  // __macro(cudnnSpatialTfGridGeneratorBackward);     
  // __macro(cudnnSpatialTfSamplerForward);            
  // __macro(cudnnSpatialTfSamplerBackward);           
  // __macro(cudnnTransformTensor);
  // __macro(cudnnSetTensorNdDescriptorEx);

HIPDNN_DNN_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP)


#define HIPDNN_DNN_ROUTINE_EACH_R2(__macro) \
  __macro(hipdnnAddTensor);                 \
  __macro(hipdnnConvolutionBackwardData);   \
  __macro(hipdnnConvolutionBackwardFilter);
HIPDNN_DNN_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP)

// APIs available after R3:
#define HIPDNN_DNN_ROUTINE_EACH_AFTER_R3(__macro)           \
  __macro(hipdnnGetConvolutionBackwardFilterWorkspaceSize); \
  __macro(hipdnnGetConvolutionBackwardDataWorkspaceSize);
HIPDNN_DNN_ROUTINE_EACH_AFTER_R3(DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP)

// APIs available after R3:
#define HIPDNN_DNN_ROUTINE_EACH_AFTER_R3_LESS_R8(__macro) \
  __macro(hipdnnGetConvolutionBackwardFilterAlgorithm);   \
  __macro(hipdnnGetConvolutionForwardAlgorithm);          \
  __macro(hipdnnGetConvolutionBackwardDataAlgorithm);     \
  __macro(hipdnnSetRNNDescriptor);
HIPDNN_DNN_ROUTINE_EACH_AFTER_R3_LESS_R8(DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP)

// APIs available after R4:
#define HIPDNN_DNN_ROUTINE_EACH_AFTER_R4(__macro)    \
  __macro(hipdnnBatchNormalizationForwardTraining);  \
  __macro(hipdnnBatchNormalizationForwardInference); \
  __macro(hipdnnBatchNormalizationBackward);
HIPDNN_DNN_ROUTINE_EACH_AFTER_R4(DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP)

// APIs in R5
#define HIPDNN_DNN_ROUTINE_EACH_R5(__macro)  \
  __macro(hipdnnCreateActivationDescriptor); \
  __macro(hipdnnSetActivationDescriptor);    \
  __macro(hipdnnGetActivationDescriptor);    \
  __macro(hipdnnDestroyActivationDescriptor);
HIPDNN_DNN_ROUTINE_EACH_R5(DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP)

// APIs in R6
#define HIPDNN_DNN_ROUTINE_EACH_R6(__macro) __macro(hipdnnSetRNNDescriptor_v6);
HIPDNN_DNN_ROUTINE_EACH_R6(DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP)

// APIs in R7
#define HIPDNN_DNN_ROUTINE_EACH_R7(__macro)                \
  __macro(hipdnnSetConvolutionGroupCount);                 \
  __macro(hipdnnSetConvolutionMathType);                   \
  __macro(hipdnnCreateCTCLossDescriptor);                  \
  __macro(hipdnnDestroyCTCLossDescriptor);                 \
  __macro(hipdnnGetCTCLossDescriptor);                     \
  __macro(hipdnnSetCTCLossDescriptor);                     \
  __macro(hipdnnGetCTCLossWorkspaceSize);                  \
  __macro(hipdnnCTCLoss);

// HIP platform not supported, refer to the following url:
// https://github.com/ROCm-Developer-Tools/HIP/blob/roc-3.5.x/docs/markdown/CUDNN_API_supported_by_HIP.md
  // __macro(cudnnConvolutionBiasActivationForward);
  // __macro(cudnnGetConvolutionBackwardDataAlgorithm_v7);
  // __macro(cudnnGetConvolutionBackwardFilterAlgorithm_v7);
  // __macro(cudnnGetConvolutionForwardAlgorithm_v7);
  // __macro(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);

HIPDNN_DNN_ROUTINE_EACH_R7(DECLARE_DYNAMIC_LOAD_HIPDNN_WRAP)


// HIP platform not supported, refer to the following url:
// https://github.com/ROCm-Developer-Tools/HIP/blob/roc-3.5.x/docs/markdown/CUDNN_API_supported_by_HIP.md
  // __macro(cudnnCreateRNNDataDescriptor); 
  // __macro(cudnnDestroyRNNDataDescriptor);
  // __macro(cudnnSetRNNDataDescriptor);    
  // __macro(cudnnSetRNNPaddingMode);       
  // __macro(cudnnRNNForwardTrainingEx);    
  // __macro(cudnnRNNBackwardDataEx);       
  // __macro(cudnnRNNBackwardWeightsEx);    
  // __macro(cudnnRNNForwardInferenceEx);
  // __macro(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize);
  // __macro(cudnnBatchNormalizationForwardTrainingEx);                
  // __macro(cudnnGetBatchNormalizationBackwardExWorkspaceSize);       
  // __macro(cudnnBatchNormalizationBackwardEx);                       
  // __macro(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
  // __macro(cudnnSetRNNDescriptor_v8);

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
