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

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(elementwise_add);
USE_OP(elementwise_add_grad);

namespace paddle {
namespace operators {

template <typename T>
static void print_data(const T * data, const int64_t numel, const std::vector<int64_t> dims, const std::string name) {
  printf("------------%s------------\n", name.c_str());
  size_t stride = dims[dims.size() - 1];
  size_t index = 0;
  while(index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", data[index]);
    } else {
      printf("%d ", data[index]);
    }
    if((index+1) % stride == 0) printf("\n");
    index ++;
  }
}

// feed data
template <typename T>
static void feed_tensor_data(const platform::DeviceContext& ctx, 
                             const framework::DDim dims,
                             framework::LoDTensor* tensor,
                             const bool feed_ones,
                             const std::string name) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  if (feed_ones) {
    for (size_t i = 0; i < numel; ++i) {
      data[i] = static_cast<T>(1);
    }
  } else {
    for (size_t i = 0; i < numel; ++i) {
      data[i] = static_cast<T>(i);
    }
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
  print_data(data.data(), numel, framework::vectorize(dims), name);
}

// input
const std::vector<int64_t> x_dim_vec = {2, 3, 4};
const int64_t x_numel = std::accumulate(x_dim_vec.begin(), x_dim_vec.end(), 1, std::multiplies<int64_t>());
const std::vector<int64_t> y_dim_vec = {4};
const int64_t y_numel = std::accumulate(y_dim_vec.begin(), y_dim_vec.end(), 1, std::multiplies<int64_t>());
// attrs
const int axis = -1;

template <typename T>
void TestMain(const platform::DeviceContext& ctx, 
              std::vector<float>& output,
              std::vector<float>& x_grad,
              std::vector<float>& y_grad) {
  auto place = ctx.GetPlace();
  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim x_dims = framework::make_ddim(x_dim_vec);
  framework::DDim y_dims = framework::make_ddim(y_dim_vec);

  // --------------- forward ----------------------
  desc_fwd.SetType("elementwise_add");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetInput("Y", {"Y"});
  desc_fwd.SetOutput("Out", {"Out"});
  desc_fwd.SetAttr("axis", axis);
  desc_fwd.SetAttr("use_mkldnn", false);

  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto y_tensor = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  auto out_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();

  // feed input data
  feed_tensor_data<T>(ctx, x_dims, x_tensor, true, "X");
  feed_tensor_data<T>(ctx, y_dims, y_tensor, true, "Y");

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*out_tensor, ctx, &output);

  // // --------------- backward ----------------------
  desc_bwd.SetType("elementwise_add_grad");
  desc_bwd.SetInput("X", {"X"});
  desc_bwd.SetInput("Y", {"Y"});
  desc_bwd.SetInput(framework::GradVarName("Out"), {framework::GradVarName("Out")});
  desc_bwd.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc_bwd.SetOutput(framework::GradVarName("Y"), {framework::GradVarName("Y")});
  desc_bwd.SetAttr("axis", axis);
  desc_bwd.SetAttr("use_mkldnn", false);

  auto out_grad_tensor = scope.Var(framework::GradVarName("Out"))->GetMutable<framework::LoDTensor>();
  auto x_grad_tensor = scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();
  auto y_grad_tensor = scope.Var(framework::GradVarName("Y"))->GetMutable<framework::LoDTensor>();

  // feed loss_grad_tensor data
  feed_tensor_data<T>(ctx, out_tensor->dims(), out_grad_tensor, true, "out_grad");

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*x_grad_tensor, ctx, &x_grad);
  framework::TensorToVector(*y_grad_tensor, ctx, &y_grad);
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
    print_data(cpu_out.data(), numel, dims, name + "_cpu");
    print_data(gpu_out.data(), numel, dims, name + "_gpu");
  }
}

TEST(test_add_op, compare_cpu_and_gpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> output_cpu(x_numel);
  std::vector<float> x_grad_cpu(x_numel);
  std::vector<float> y_grad_cpu(y_numel);
  TestMain<float>(cpu_ctx, output_cpu, x_grad_cpu, y_grad_cpu);

  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> output_gpu(x_numel);
  std::vector<float> x_grad_gpu(x_numel);
  std::vector<float> y_grad_gpu(y_numel);
  TestMain<float>(gpu_ctx, output_gpu, x_grad_gpu, y_grad_gpu);

  compare_results<float>(output_cpu, output_gpu, x_numel, x_dim_vec, "Out");
  compare_results<float>(x_grad_cpu, x_grad_gpu, x_numel, x_dim_vec, "XGrad");
  compare_results<float>(y_grad_cpu, x_grad_gpu, y_numel, y_dim_vec, "YGrad");
}

}  // namespace operators
}  // namespace paddle