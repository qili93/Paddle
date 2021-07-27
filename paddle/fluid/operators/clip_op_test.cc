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

USE_OP(clip);
USE_OP(clip_grad);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
USE_OP_DEVICE_KERNEL(clip, CUDA);
USE_OP_DEVICE_KERNEL(clip_grad, CUDA);
#endif

#ifdef PADDLE_WITH_ASCEND_CL
USE_OP_DEVICE_KERNEL(clip, NPU);
USE_OP_DEVICE_KERNEL(clip_grad, NPU);
#endif

namespace paddle {
namespace operators {

template <typename T>
static void print_data(const T* data, const int64_t numel, const std::string name) {
  printf("%s = [ ", name.c_str());
  for (int64_t i = 0; i < numel; ++i) {
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
                      const framework::DDim dims,
                      framework::LoDTensor* tensor,
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
                       const framework::DDim dims,
                       framework::LoDTensor* tensor,
                       const T value) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(value + i);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

const bool set_input = false;
const float min = 1.5;
const float max = 7.5;
const std::vector<int64_t> x_dims = {2, 5};
const int64_t x_numel = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int64_t>());

void SetMinMaxDesc(framework::OpDesc& op_desc, bool has_input) {
  if (has_input) {
    op_desc.SetInput("Min", {"Min"});
    op_desc.SetInput("Max", {"Max"});
  } else {
    op_desc.SetInput("Min", {});
    op_desc.SetInput("Max", {});
  }
}

template <typename T>
void feedMinMaxInput(const platform::DeviceContext& ctx, framework::Scope& scope, bool has_input) {
  if (has_input) {
    auto min_tensor = scope.Var("Min")->GetMutable<framework::LoDTensor>();
    auto max_tensor = scope.Var("Max")->GetMutable<framework::LoDTensor>();
    feed_value<T>(ctx, framework::make_ddim({1}), min_tensor, static_cast<T>(1.5));
    feed_value<T>(ctx, framework::make_ddim({1}), max_tensor, static_cast<T>(7.5));
  }
}

template <typename T>
void TestMain(const platform::DeviceContext& ctx,
              std::vector<float>& out_data,
              std::vector<float>& x_grad_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim x_ddims = framework::make_ddim(x_dims);

  // --------------- forward ----------------------
  desc_fwd.SetType("clip");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetOutput("Out", {"Out"});
  // desc_fwd.SetInput("Min", {});
  // desc_fwd.SetInput("Max", {});
  // desc_fwd.SetInput("Min", {"Min"});
  // desc_fwd.SetInput("Max", {"Max"});
  desc_fwd.SetAttr("min", min);
  desc_fwd.SetAttr("max", max);
  SetMinMaxDesc(desc_fwd, set_input);
  feedMinMaxInput<T>(ctx, scope, set_input);

  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  // auto min_tensor = scope.Var("Min")->GetMutable<framework::LoDTensor>();
  // auto max_tensor = scope.Var("Max")->GetMutable<framework::LoDTensor>();
  auto out_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();

  feed_range<T>(ctx, x_ddims, x_tensor, static_cast<T>(0.0));
  // feed_value<T>(ctx, framework::make_ddim({1}), min_tensor, static_cast<T>(1.5));
  // feed_value<T>(ctx, framework::make_ddim({1}), max_tensor, static_cast<T>(7.5));

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*out_tensor, ctx, &out_data);

  // --------------- backward ----------------------
  desc_bwd.SetType("clip_grad");
  desc_bwd.SetInput("X", {"X"});
  // desc_bwd.SetInput("Min", {"Min"});
  // desc_bwd.SetInput("Max", {"Max"});
  // desc_bwd.SetInput("Min", {});
  // desc_bwd.SetInput("Max", {});
  desc_bwd.SetInput(framework::GradVarName("Out"), {framework::GradVarName("Out")});
  desc_bwd.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc_bwd.SetAttr("min", min);
  desc_bwd.SetAttr("max", max);
  SetMinMaxDesc(desc_bwd, set_input);

  auto out_grad_tensor = scope.Var(framework::GradVarName("Out"))->GetMutable<framework::LoDTensor>();
  auto x_grad_tensor = scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();
  feed_value<T>(ctx, x_ddims, out_grad_tensor, static_cast<T>(1.0));

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*x_grad_tensor, ctx, &x_grad_data);
}

template <typename T>
static void compare_results(const std::vector<T> cpu_data,
                            const std::vector<T> npu_data,
                            const int64_t data_numel,
                            const std::string name) {
  auto result = std::equal(
      cpu_data.begin(), cpu_data.end(), npu_data.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-9; });
  if (!result) {
    LOG(INFO) << "=========== Ouptut " << name << " is NOT Equal !!!!! ===========";
    print_data(cpu_data.data(), data_numel, name + "_cpu");
    print_data(npu_data.data(), data_numel, name + "_dev");
  } else {
    LOG(INFO) << "=========== Ouptut " << name << " is Equal in CPU and GPU ===========";
    print_data(cpu_data.data(), data_numel, name + "_cpu");
    print_data(npu_data.data(), data_numel, name + "_dev");
  }
}

TEST(test_stack_op, compare_cpu_and_npu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> cpu_out_data(x_numel);
  std::vector<float> cpu_x_grad_data(x_numel);
  TestMain<float>(cpu_ctx, cpu_out_data, cpu_x_grad_data);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  platform::CUDAPlace gpu_place;
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> gpu_out_data(x_numel);
  std::vector<float> gpu_x_grad_data(x_numel);
  TestMain<float>(gpu_ctx, gpu_out_data, gpu_x_grad_data);

  compare_results<float>(cpu_out_data, gpu_out_data, x_numel, "Out_gpu");
  compare_results<float>(cpu_x_grad_data, gpu_x_grad_data, x_numel, "X@Grad_gpu");
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> npu_out_data(x_numel);
  std::vector<float> npu_x_grad_data(x_numel);
  TestMain<float>(npu_ctx, npu_out_data, npu_x_grad_data);

  compare_results<float>(cpu_out_data, npu_out_data, x_numel, "Out_npu");
  compare_results<float>(cpu_x_grad_data, npu_x_grad_data, x_numel, "X@Grad_npu");
#endif
}

}  // namespace operators
}  // namespace paddle