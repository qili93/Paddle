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

USE_OP(leaky_relu);
USE_OP(leaky_relu_grad);

#ifdef PADDLE_WITH_ASCEND_CL
USE_OP_DEVICE_KERNEL(leaky_relu, NPU);
USE_OP_DEVICE_KERNEL(leaky_relu_grad, NPU);
#endif

namespace paddle {
namespace operators {

template <typename T>
static void print_data(const T* data, const size_t& numel,
                       const std::string& name) {
  printf("%s = [ ", name.c_str());
  for (size_t i = 0; i < numel; ++i) {
    if (std::is_floating_point<T>::value) {
      printf("%.1f, ", static_cast<float>(data[i]));
    } else {
      printf("%d, ", static_cast<int>(data[i]));
    }
  }
  printf("]\n");
}

template <typename T>
static void feed_value(const platform::DeviceContext& ctx,
                       const framework::DDim dims, framework::LoDTensor* tensor,
                       const T value) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(value);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

template <typename T>
static void feed_range(const platform::DeviceContext& ctx,
                       const framework::DDim dims, framework::LoDTensor* tensor,
                       const T value) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(value + i);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

template <typename T>
void TestMain(const platform::DeviceContext& ctx, std::vector<T>* out_data,
              std::vector<T>* x_grad_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  const std::vector<int64_t> x_dims{2, 4};

  framework::DDim x_ddims = framework::make_ddim(x_dims);
  framework::DDim out_ddims = framework::make_ddim(x_dims);

  // --------------- forward ----------------------
  desc_fwd.SetType("leaky_relu");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetOutput("Out", {"Out"});
  desc_fwd.SetAttr("alpha", static_cast<float>(0.1));

  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto out_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();

  feed_range<T>(ctx, x_ddims, x_tensor, static_cast<T>(-4.0));

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*out_tensor, ctx, out_data);

  // --------------- backward ----------------------
  desc_bwd.SetType("leaky_relu_grad");
  desc_bwd.SetInput("X", {"X"});
  desc_bwd.SetInput(framework::GradVarName("Out"),
                    {framework::GradVarName("Out")});
  desc_bwd.SetOutput(framework::GradVarName("X"),
                     {framework::GradVarName("X")});
  desc_bwd.SetAttr("alpha", static_cast<float>(0.1));
  desc_bwd.SetAttr("use_mkldnn", false);

  auto out_grad_tensor = scope.Var(framework::GradVarName("Out"))
                             ->GetMutable<framework::LoDTensor>();
  auto x_grad_tensor = scope.Var(framework::GradVarName("X"))
                           ->GetMutable<framework::LoDTensor>();
  feed_value<T>(ctx, out_ddims, out_grad_tensor, static_cast<T>(1.0));

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*x_grad_tensor, ctx, x_grad_data);
}

template <typename T>
static void compare_results(const std::vector<T>& cpu_data,
                            const std::vector<T>& dev_data,
                            const std::string& name) {
  auto result = std::equal(
      cpu_data.begin(), cpu_data.end(), dev_data.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-9; });
  if (!result) {
    LOG(INFO) << "=========== " << name
              << " is NOT Equal !!!!!!!!! ===========";
    print_data<T>(cpu_data.data(), cpu_data.size(), name + "_cpu");
    print_data<T>(dev_data.data(), dev_data.size(), name + "_dev");
  } else {
    LOG(INFO) << "=========== " << name
              << " is Equal in CPU and GPU ===========";
    print_data<T>(cpu_data.data(), cpu_data.size(), name + "_cpu");
    print_data<T>(dev_data.data(), dev_data.size(), name + "_dev");
  }
}

TEST(test_interpolate_op, compare_cpu_and_dev) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> cpu_out_data(8);
  std::vector<float> cpu_grad_data(8);
  TestMain<float>(cpu_ctx, &cpu_out_data, &cpu_grad_data);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  platform::CUDAPlace gpu_place;
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> gpu_out_data(8);
  std::vector<float> gpu_grad_data(8);
  TestMain<float>(gpu_ctx, &gpu_out_data, &gpu_grad_data);

  compare_results<float>(cpu_out_data, gpu_out_data, "output");
  compare_results<float>(cpu_grad_data, gpu_grad_data, "x_grad");
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> npu_out_data(8);
  std::vector<float> npu_grad_data(8);
  TestMain<float>(npu_ctx, &npu_out_data, &npu_grad_data);

  compare_results<float>(cpu_out_data, npu_out_data, "output");
  compare_results<float>(cpu_grad_data, npu_grad_data, "x_grad");
#endif
}

}  // namespace operators
}  // namespace paddle
