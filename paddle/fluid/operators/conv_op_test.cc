// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(depthwise_conv2d);
USE_OP_DEVICE_KERNEL(depthwise_conv2d, CUDNN);
USE_OP(depthwise_conv2d_grad);
USE_OP_DEVICE_KERNEL(depthwise_conv2d_grad, CUDNN);

namespace paddle {
namespace operators {

template <typename T>
static void print_data(const T* data, const int64_t numel,
                       const std::vector<int64_t> dims,
                       const std::string name) {
  printf("------------%s------------\n", name.c_str());
  size_t stride_idx = dims.size() - 1;
  size_t stride = dims[stride_idx];
  while (stride < 2) {
    stride_idx--;
    stride = dims[stride_idx];
  }
  stride_idx--;
  size_t stride2 = stride_idx > 0 ? dims[stride_idx] : numel;
  while (stride2 < 2) {
    stride_idx--;
    stride2 = dims[stride_idx];
  }
  size_t index = 0;
  while (index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", static_cast<float>(data[index]));
    } else {
      printf("%d ", static_cast<int>(data[index]));
    }
    if ((index + 1) % stride == 0) printf("\n");
    if ((index + 1) % stride2 == 0) printf("\n");
    index++;
  }
}

// feed data
template <typename T>
static void feed_ones(const platform::DeviceContext& ctx,
                      const framework::DDim dims,
                      framework::LoDTensor* tensor,
                      const std::string name) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(1);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
  //   print_data(data.data(), numel, framework::vectorize(dims), name);
}

// input
const int64_t batch_size = 2;
const int64_t input_channel = 3;
const int64_t input_height = 5;
const int64_t input_width = 5;
const std::vector<int64_t> input_dim_vec = {batch_size, input_height, input_width, input_channel}; // NHWC
const int64_t input_numel = std::accumulate(input_dim_vec.begin(), input_dim_vec.end(), 1, std::multiplies<int64_t>());
const int64_t out_channel = 12;
const int64_t kernel_h = 3;
const int64_t kernel_w = 3;
const int groups = 3;
const std::vector<int64_t> filter_dim_vec = {out_channel, 1, kernel_h, kernel_w};
const int64_t filter_numel = std::accumulate(filter_dim_vec.begin(), filter_dim_vec.end(), 1, std::multiplies<int64_t>());
// attrs
const std::vector<int> strides = {2, 2};
const std::vector<int> paddings = {2, 1, 2, 3};
const std::vector<int> dilations = {1, 1};
// attrs
const bool exhaustive_search = false;
const std::string padding_algorithm = "EXPLICIT";
const std::string data_format = "NHWC";
const bool use_addto = false;
// output
const int64_t out_height = (input_height + paddings[0] + paddings[1]- (dilations[0] * (kernel_h - 1) + 1)) / strides[0] + 1;
const int64_t out_width = (input_width + paddings[2] + paddings[3]- (dilations[1] * (kernel_w - 1) + 1)) / strides[1] + 1;
const std::vector<int64_t> out_dim_vec = {batch_size, out_height, out_width, out_channel};  // NHWC
const int64_t out_numel = std::accumulate(out_dim_vec.begin(), out_dim_vec.end(), 1, std::multiplies<int64_t>());

template <typename T>
void TestMain(const platform::DeviceContext& ctx,
              std::vector<float>& out,
              std::vector<float>& input_grad,
              std::vector<float>& filter_grad) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim input_dims = framework::make_ddim(input_dim_vec);
  framework::DDim filter_dims = framework::make_ddim(filter_dim_vec);
  framework::DDim out_dims = framework::make_ddim(out_dim_vec);

  LOG(INFO) << "input_dims=" << input_dims.to_str();
  LOG(INFO) << "filter_dims=" << filter_dims.to_str();
  LOG(INFO) << "out_dims=" << out_dims.to_str();

  // --------------- forward ----------------------
  desc_fwd.SetType("depthwise_conv2d");
  desc_fwd.SetInput("Input", {"Input"});
  desc_fwd.SetInput("Filter", {"Filter"});
  desc_fwd.SetOutput("Output", {"Output"});
  desc_fwd.SetAttr("groups", groups);
  desc_fwd.SetAttr("strides", strides);
  desc_fwd.SetAttr("paddings", paddings);
  desc_fwd.SetAttr("dilations", dilations);
  desc_fwd.SetAttr("exhaustive_search", exhaustive_search);
  desc_fwd.SetAttr("padding_algorithm", padding_algorithm);
  desc_fwd.SetAttr("data_format", data_format);
  desc_fwd.SetAttr("use_addto", use_addto);
  desc_fwd.SetAttr("use_cudnn", true);
  desc_fwd.SetAttr("use_mkldnn", false);

  auto input_tensor = scope.Var("Input")->GetMutable<framework::LoDTensor>();
  auto filter_tensor = scope.Var("Filter")->GetMutable<framework::LoDTensor>();
  auto out_tensor = scope.Var("Output")->GetMutable<framework::LoDTensor>();

  // feed data to input tensors
  feed_ones<T>(ctx, input_dims, input_tensor, "Input");
  feed_ones<T>(ctx, filter_dims, filter_tensor, "Filter");

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*out_tensor, ctx, &out);

  // --------------- backward ----------------------
  desc_bwd.SetType("depthwise_conv2d_grad");
  desc_bwd.SetInput("Input", {"Input"});
  desc_bwd.SetInput("Filter", {"Filter"});
  desc_bwd.SetInput(framework::GradVarName("Output"), {framework::GradVarName("Output")});
  desc_bwd.SetOutput(framework::GradVarName("Input"), {framework::GradVarName("Input")});
  desc_bwd.SetOutput(framework::GradVarName("Filter"), {framework::GradVarName("Filter")});
  desc_bwd.SetAttr("groups", groups);
  desc_bwd.SetAttr("strides", strides);
  desc_bwd.SetAttr("paddings", paddings);
  desc_bwd.SetAttr("dilations", dilations);
  desc_bwd.SetAttr("exhaustive_search", exhaustive_search);
  desc_bwd.SetAttr("padding_algorithm", padding_algorithm);
  desc_bwd.SetAttr("data_format", data_format);
  desc_bwd.SetAttr("use_addto", use_addto);
  desc_bwd.SetAttr("use_cudnn", true);
  desc_bwd.SetAttr("use_mkldnn", false);

  auto out_grad_tensor = scope.Var(framework::GradVarName("Output"))->GetMutable<framework::LoDTensor>();
  auto input_grad_tensor = scope.Var(framework::GradVarName("Input"))->GetMutable<framework::LoDTensor>();
  auto filter_grad_tensor = scope.Var(framework::GradVarName("Filter"))->GetMutable<framework::LoDTensor>();

  // feed data to input tensors
  feed_ones<T>(ctx, out_dims, out_grad_tensor, "Output_Grad");

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*input_grad_tensor, ctx, &input_grad);
  framework::TensorToVector(*filter_grad_tensor, ctx, &filter_grad);
}

template <typename T>
static void compare_results(const std::vector<T> cpu_out,
                            const std::vector<T> gpu_out, const int64_t numel,
                            const std::vector<int64_t> dims,
                            const std::string name) {
  auto result = std::equal(
      cpu_out.begin(), cpu_out.end(), gpu_out.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-9; });
  if (!result) {
    LOG(INFO) << "=========== Ouptut " << name << " is NOT Equal !!!!! ===========";
    print_data(cpu_out.data(), numel, dims, name + "_cpu");
    print_data(gpu_out.data(), numel, dims, name + "_gpu");
  } else {
    LOG(INFO) << "=========== Ouptut " << name << " is Equal in CPU and GPU ===========";
    // print_data(cpu_out.data(), numel, dims, name + "_cpu");
    // print_data(gpu_out.data(), numel, dims, name + "_gpu");
  }
}

TEST(test_pool2d_op, compare_cpu_and_gpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> out_cpu(out_numel);
  std::vector<float> input_grad_cpu(input_numel);
  std::vector<float> filter_grad_cpu(filter_numel);
  TestMain<float>(cpu_ctx, out_cpu, input_grad_cpu, filter_grad_cpu);

  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> out_gpu(out_numel);
  std::vector<float> input_grad_gpu(input_numel);
  std::vector<float> filter_grad_gpu(input_numel);
  TestMain<float>(gpu_ctx, out_gpu, input_grad_gpu, filter_grad_gpu);

  compare_results<float>(out_cpu, out_gpu, out_numel, out_dim_vec, "output");
  compare_results<float>(input_grad_cpu, input_grad_gpu, input_numel, input_dim_vec, "input_grad");
  compare_results<float>(filter_grad_cpu, filter_grad_gpu, filter_numel, filter_dim_vec, "filter_grad");
}

}  // namespace operators
}  // namespace paddle