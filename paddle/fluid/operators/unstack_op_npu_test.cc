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

USE_OP(unstack);
USE_OP(unstack_grad);

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

const int num = 3;
const int axis = 0;
const std::vector<int64_t> x_dims = {3, 1, 2};
const std::vector<int64_t> y_dims = {1, 2};
const int64_t x_numel = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int64_t>());
const int64_t y_numel = std::accumulate(y_dims.begin(), y_dims.end(), 1, std::multiplies<int64_t>());

template <typename T>
void TestMain(const platform::DeviceContext& ctx,
              std::vector<float>& y0_data,
              std::vector<float>& y1_data,
              std::vector<float>& y2_data,
              std::vector<float>& x_grad_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim x_ddims = framework::make_ddim(x_dims);
  framework::DDim y_ddims = framework::make_ddim(y_dims);

  // --------------- forward ----------------------
  desc_fwd.SetType("unstack");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetOutput("Y", {"y0", "y1", "y2"});
  desc_fwd.SetAttr("num", num);
  desc_fwd.SetAttr("axis", axis);

  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  feed_ones<T>(ctx, x_ddims, x_tensor, "X");

  auto y0_tensor = scope.Var("y0")->GetMutable<framework::LoDTensor>();
  auto y1_tensor = scope.Var("y1")->GetMutable<framework::LoDTensor>();
  auto y2_tensor = scope.Var("y2")->GetMutable<framework::LoDTensor>();

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*y0_tensor, ctx, &y0_data);
  framework::TensorToVector(*y1_tensor, ctx, &y1_data);
  framework::TensorToVector(*y2_tensor, ctx, &y2_data);

  // --------------- backward ----------------------
  desc_bwd.SetType("unstack_grad");
  desc_bwd.SetInput(framework::GradVarName("Y"), {"y0_grad", "y1_grad", "y2_grad"});
  desc_bwd.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc_bwd.SetAttr("num", num);
  desc_bwd.SetAttr("axis", axis);

  auto y0_grad_tensor = scope.Var("y0_grad")->GetMutable<framework::LoDTensor>();
  auto y1_grad_tensor = scope.Var("y1_grad")->GetMutable<framework::LoDTensor>();
  auto y2_grad_tensor = scope.Var("y2_grad")->GetMutable<framework::LoDTensor>();
  feed_ones<T>(ctx, y_ddims, y0_grad_tensor, "y0_grad");
  feed_ones<T>(ctx, y_ddims, y1_grad_tensor, "y1_grad");
  feed_ones<T>(ctx, y_ddims, y2_grad_tensor, "y2_grad");

  auto x_grad_tensor = scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();

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
  std::vector<float> cpu_y0_data(y_numel);
  std::vector<float> cpu_y1_data(y_numel);
  std::vector<float> cpu_y2_data(y_numel);
  std::vector<float> cpu_x_grad_data(x_numel);
  TestMain<float>(cpu_ctx, cpu_y0_data, cpu_y1_data, cpu_y2_data, cpu_x_grad_data);

  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> npu_y0_data(y_numel);
  std::vector<float> npu_y1_data(y_numel);
  std::vector<float> npu_y2_data(y_numel);
  std::vector<float> npu_x_grad_data(x_numel);
  TestMain<float>(npu_ctx, npu_y0_data, npu_y1_data, npu_y2_data, npu_x_grad_data);

  compare_results<float>(cpu_y0_data, npu_y0_data, y_numel, "y0");
  compare_results<float>(cpu_y1_data, npu_y1_data, y_numel, "y1");
  compare_results<float>(cpu_y2_data, npu_y2_data, y_numel, "y2");
  compare_results<float>(cpu_x_grad_data, npu_x_grad_data, x_numel, "x_grad");
}

}  // namespace operators
}  // namespace paddle