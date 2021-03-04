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

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(dropout);
USE_OP(dropout_grad);

namespace paddle {
namespace operators {

template <typename T>
static void feed_tensor_data(const platform::DeviceContext& ctx, 
                             const framework::DDim dims,
                             framework::LoDTensor* tensor) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = 1;
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

template <typename T>
static void print_data(const T * data, const int64_t numel, const std::vector<int64_t> dims, const std::string name) {
  printf("------------%s------------\n", name.c_str());
  size_t stride_idx = dims.size() - 1;
  size_t stride = dims[stride_idx];
  while(stride < 2) {
    stride_idx--;
    stride = dims[stride_idx];
  }
  size_t index = 0;
  while(index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", static_cast<float>(data[index]));
    } else {
      printf("%d ", static_cast<int>(data[index]));
    }
    if((index+1) % stride == 0) printf("\n");
    index ++;
  }
}

// OP attrs
const int input_height = 32;
const int input_width = 64;
const std::vector<int64_t> input_dim_vec = {input_height, input_width};
const int input_numel = input_height * input_width;
const float dropout_prob = 0.0;
const std::string dropout_implementation = "downgrade_in_infer";
const bool fix_seed = true;
const int seed = 0;

template <typename T>
void TestDropOut(const platform::DeviceContext& ctx, std::vector<float>& output, std::vector<uint8_t>& mask) {
  auto place = ctx.GetPlace();
  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim input_dims({input_height, input_width});

  // --------------- forward ----------------------
  desc_fwd.SetType("dropout");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetOutput("Out", {"Out"});
  desc_fwd.SetOutput("Mask", {"Mask"});
  desc_fwd.SetAttr("dropout_prob", dropout_prob);
  desc_fwd.SetAttr("dropout_implementation", dropout_implementation);
  desc_fwd.SetAttr("is_test", false);
  desc_fwd.SetAttr("fix_seed", fix_seed);
  desc_fwd.SetAttr("seed", seed);
  desc_fwd.SetAttr("use_cudnn", false);
  desc_fwd.SetAttr("use_mkldnn", false);

  auto input_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto output_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();
  auto mask_tensor = scope.Var("Mask")->GetMutable<framework::LoDTensor>();

  // feed input data
  feed_tensor_data<T>(ctx, input_dims, input_tensor);

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  // get output
  framework::TensorToVector(*output_tensor, ctx, &output);
  framework::TensorToVector(*mask_tensor, ctx, &mask);
}

template <typename T>
static void compare_results(const std::vector<T> cpu_out, 
                            const std::vector<T> gpu_out, 
                            const int64_t numel, 
                            const std::vector<int64_t> dims, 
                            const std::string name) {
  auto result = std::equal(
      cpu_out.begin(), cpu_out.end(), gpu_out.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-4; });
  if (!result) {
    LOG(INFO) << "=========== Ouptut " << name << " is NOT Equal ===========";
    print_data(cpu_out.data(), numel, dims, name + "_cpu");
    print_data(gpu_out.data(), numel, dims, name + "_gpu");
  } else {
    LOG(INFO) << "=========== Ouptut " << name << " is Equal in CPU and GPU ===========";
    // print_data(cpu_out.data(), numel, dims, name + "_cpu");
    // print_data(gpu_out.data(), numel, dims, name + "_gpu");
  }
}

TEST(test_dropout_op, cpu_place) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> output_cpu(input_numel);
  std::vector<uint8_t> mask_cpu(input_numel);
  TestDropOut<float>(cpu_ctx, output_cpu, mask_cpu);

  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> output_gpu(input_numel);
  std::vector<uint8_t> mask_gpu(input_numel);
  TestDropOut<float>(gpu_ctx, output_gpu, mask_gpu);

  compare_results<float>(output_cpu, output_gpu, input_numel, input_dim_vec, "output");
  compare_results<uint8_t>(mask_cpu, mask_gpu, input_numel, input_dim_vec, "mask");
}

}  // namespace operators
}  // namespace paddle