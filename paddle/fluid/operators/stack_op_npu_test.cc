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

USE_OP(stack);
USE_OP(stack_grad);

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
  //   print_data(data.data(), numel, name);
}

template <typename T>
static void feed_range(const platform::DeviceContext& ctx,
                       const framework::DDim dims,
                       framework::LoDTensor* tensor,
                       const std::string name,
                       const T value) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(value + i);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
  //   print_data(data.data(), numel, name);
}

const int num = 3;
const int axis = -2;
const std::vector<int64_t> x_dims = {1, 2};
const std::vector<int64_t> y_dims = {1, 3, 2};
const int64_t x_numel = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int64_t>());
const int64_t y_numel = std::accumulate(y_dims.begin(), y_dims.end(), 1, std::multiplies<int64_t>());

template <typename T>
void TestMain(const platform::DeviceContext& ctx,
              std::vector<float>& y_data,
              std::vector<float>& x0_grad_data,
              std::vector<float>& x1_grad_data,
              std::vector<float>& x2_grad_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim x_ddims = framework::make_ddim(x_dims);
  framework::DDim y_ddims = framework::make_ddim(y_dims);

  // --------------- forward ----------------------
  desc_fwd.SetType("stack");
  desc_fwd.SetInput("X", {"x0", "x1", "x2"});
  desc_fwd.SetOutput("Y", {"Y"});
  desc_fwd.SetAttr("num", num);
  desc_fwd.SetAttr("axis", axis);

  auto x0_tensor = scope.Var("x0")->GetMutable<framework::LoDTensor>();
  auto x1_tensor = scope.Var("x1")->GetMutable<framework::LoDTensor>();
  auto x2_tensor = scope.Var("x2")->GetMutable<framework::LoDTensor>();

  feed_range<T>(ctx, x_ddims, x0_tensor, "x0", static_cast<T>(1.0));
  feed_range<T>(ctx, x_ddims, x1_tensor, "x1", static_cast<T>(3.0));
  feed_range<T>(ctx, x_ddims, x2_tensor, "x2", static_cast<T>(5.0));
  auto y_tensor = scope.Var("Y")->GetMutable<framework::LoDTensor>();

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*y_tensor, ctx, &y_data);

  // --------------- backward ----------------------
  desc_bwd.SetType("stack_grad");
  desc_bwd.SetInput(framework::GradVarName("Y"), {framework::GradVarName("Y")});
  desc_bwd.SetOutput(framework::GradVarName("X"), {"x0_grad", "x1_grad", "x2_grad"});
  desc_bwd.SetAttr("num", num);
  desc_bwd.SetAttr("axis", axis);

  auto y_grad_tensor = scope.Var(framework::GradVarName("Y"))->GetMutable<framework::LoDTensor>();
  feed_ones<T>(ctx, y_ddims, y_grad_tensor, "Y_Grad");

  auto x0_grad_tensor = scope.Var("x0_grad")->GetMutable<framework::LoDTensor>();
  auto x1_grad_tensor = scope.Var("x1_grad")->GetMutable<framework::LoDTensor>();
  auto x2_grad_tensor = scope.Var("x2_grad")->GetMutable<framework::LoDTensor>();

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*x0_grad_tensor, ctx, &x0_grad_data);
  framework::TensorToVector(*x1_grad_tensor, ctx, &x1_grad_data);
  framework::TensorToVector(*x2_grad_tensor, ctx, &x2_grad_data);
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
    print_data(npu_data.data(), data_numel, name + "_gpu");
  } else {
    LOG(INFO) << "=========== Ouptut " << name << " is Equal in CPU and GPU ===========";
    print_data(cpu_data.data(), data_numel, name + "_cpu");
    print_data(npu_data.data(), data_numel, name + "_gpu");
  }
}

TEST(test_stack_op, compare_cpu_and_npu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> cpu_y_data(y_numel);
  std::vector<float> cpu_x0_grad_data(x_numel);
  std::vector<float> cpu_x1_grad_data(x_numel);
  std::vector<float> cpu_x2_grad_data(x_numel);
  TestMain<float>(cpu_ctx, cpu_y_data, cpu_x0_grad_data, cpu_x1_grad_data, cpu_x2_grad_data);

  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> npu_y_data(y_numel);
  std::vector<float> npu_x0_grad_data(x_numel);
  std::vector<float> npu_x1_grad_data(x_numel);
  std::vector<float> npu_x2_grad_data(x_numel);
  TestMain<float>(npu_ctx, npu_y_data, npu_x0_grad_data, npu_x1_grad_data, npu_x2_grad_data);

  compare_results<float>(cpu_y_data, npu_y_data, y_numel, "Y");
  compare_results<float>(cpu_x0_grad_data, npu_x0_grad_data, x_numel, "x0_grad");
  compare_results<float>(cpu_x1_grad_data, npu_x1_grad_data, x_numel, "x1_grad");
  compare_results<float>(cpu_x2_grad_data, npu_x2_grad_data, x_numel, "x2_grad");
}

}  // namespace operators
}  // namespace paddle