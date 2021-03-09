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

#include <fstream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(softmax_with_cross_entropy);
USE_OP(softmax_with_cross_entropy_grad);

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
  size_t index = 0;
  while (index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", static_cast<float>(data[index]));
    } else {
      printf("%d ", static_cast<int>(data[index]));
    }
    if ((index + 1) % stride == 0) printf("\n");
    index++;
  }
}

// read data
template <typename T>
static void read_tensor_data(const platform::DeviceContext& ctx,
                             const framework::DDim dims,
                             framework::LoDTensor* tensor,
                             const std::string name,
                             const std::string filename) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data;
  std::ifstream fin(filename.c_str());
  for (int i = 0; i < numel; ++i) {
    T value;
    fin >> value;
    data.push_back(value);
  }
  fin.close();
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
  //   print_data(data.data(), numel, framework::vectorize(dims), name);
}

// feed data
template <typename T>
static void feed_tensor_data(const platform::DeviceContext& ctx,
                             const framework::DDim dims,
                             framework::LoDTensor* tensor,
                             const int number_limit, const std::string name) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    // data[i] = static_cast<T>(i % number_limit);
    data[i] = static_cast<T>(1);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
  //   print_data(data.data(), numel, framework::vectorize(dims), name);
}

// input
const std::vector<int64_t> input_dim_vec = {64, 1000};
const int64_t input_numel = std::accumulate(
    input_dim_vec.begin(), input_dim_vec.end(), 1, std::multiplies<int64_t>());
const std::vector<int64_t> label_dim_vec = {64, 1};
const int64_t label_numel = std::accumulate(
    label_dim_vec.begin(), label_dim_vec.end(), 1, std::multiplies<int64_t>());
// attrs
const bool numeric_stable_mode = true;
const bool soft_label = false;
const int axis = -1;
const int ignore_index = -1;

template <typename T>
void TestCrossEntropy(const platform::DeviceContext& ctx,
                      std::vector<float>& softmax_out,
                      std::vector<float>& loss_out,
                      std::vector<float>& logit_grad_out) {
  auto place = ctx.GetPlace();
  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim input_dims = framework::make_ddim(input_dim_vec);
  framework::DDim label_dims = framework::make_ddim(label_dim_vec);

  // --------------- forward ----------------------
  desc_fwd.SetType("softmax_with_cross_entropy");
  desc_fwd.SetInput("Logits", {"Logits"});
  desc_fwd.SetInput("Label", {"Label"});
  desc_fwd.SetOutput("Softmax", {"Softmax"});
  desc_fwd.SetOutput("Loss", {"Loss"});
  desc_fwd.SetAttr("axis", axis);
  desc_fwd.SetAttr("ignore_index", ignore_index);
  desc_fwd.SetAttr("soft_label", soft_label);
  desc_fwd.SetAttr("numeric_stable_mode", numeric_stable_mode);

  auto logits_tensor = scope.Var("Logits")->GetMutable<framework::LoDTensor>();
  auto label_tensor = scope.Var("Label")->GetMutable<framework::LoDTensor>();
  auto softmax_tensor =
      scope.Var("Softmax")->GetMutable<framework::LoDTensor>();
  auto loss_tensor = scope.Var("Loss")->GetMutable<framework::LoDTensor>();

  // feed input data
  // feed_tensor_data<T>(ctx, input_dims, logits_tensor, 1000, "logit");
  // feed_tensor_data<int64_t>(ctx, label_dims, label_tensor, 10, "label");
  read_tensor_data<T>(ctx, input_dims, logits_tensor, "logit",
                      "input_data.txt");
  read_tensor_data<int64_t>(ctx, label_dims, label_tensor, "label",
                            "label_data.txt");

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*softmax_tensor, ctx, &softmax_out);
  framework::TensorToVector(*loss_tensor, ctx, &loss_out);

  //   // // --------------- backward ----------------------
  //   desc_bwd.SetType("softmax_with_cross_entropy_grad");
  //   desc_bwd.SetInput("Softmax", {"Softmax"});
  //   desc_bwd.SetInput("Label", {"Label"});
  //   desc_bwd.SetInput(framework::GradVarName("Loss"),
  //   {framework::GradVarName("Loss")});
  //   desc_bwd.SetOutput(framework::GradVarName("Logits"),
  //   {framework::GradVarName("Logits")});
  //   desc_bwd.SetAttr("axis", axis);
  //   desc_bwd.SetAttr("ignore_index", ignore_index);
  //   desc_bwd.SetAttr("soft_label", soft_label);
  //   desc_bwd.SetAttr("numeric_stable_mode", numeric_stable_mode);

  //   auto loss_grad_tensor =
  //   scope.Var(framework::GradVarName("Loss"))->GetMutable<framework::LoDTensor>();
  //   auto logit_grad_tensor =
  //   scope.Var(framework::GradVarName("Logits"))->GetMutable<framework::LoDTensor>();

  //   // feed loss_grad_tensor data
  //   feed_tensor_data<T>(ctx, loss_tensor->dims(), loss_grad_tensor, true,
  //   "loss_grad");

  //   auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  //   LOG(INFO) << op_bwd->DebugStringEx(&scope);
  //   op_bwd->Run(scope, place);
  //   platform::DeviceContextPool::Instance().Get(place)->Wait();
  //   LOG(INFO) << op_bwd->DebugStringEx(&scope);

  //   framework::TensorToVector(*logit_grad_tensor, ctx, &logit_grad_out);
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
    LOG(INFO) << "=========== Ouptut " << name << " is NOT Equal ===========";
    print_data(cpu_out.data(), numel, dims, name + "_cpu");
    print_data(gpu_out.data(), numel, dims, name + "_gpu");
  } else {
    LOG(INFO) << "=========== Ouptut " << name
              << " is Equal in CPU and GPU ===========";
    // print_data(cpu_out.data(), numel, dims, name + "_cpu");
    print_data(gpu_out.data(), numel, dims, name + "_gpu");
  }
}

TEST(test_conv2d_op, compare_cpu_and_gpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> softmax_cpu(input_numel);
  std::vector<float> loss_cpu(label_numel);
  std::vector<float> logit_grad_cpu(input_numel);
  TestCrossEntropy<float>(cpu_ctx, softmax_cpu, loss_cpu, logit_grad_cpu);

  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> softmax_gpu(input_numel);
  std::vector<float> loss_gpu(label_numel);
  std::vector<float> logit_grad_gpu(input_numel);
  TestCrossEntropy<float>(gpu_ctx, softmax_gpu, loss_gpu, logit_grad_gpu);

  //   compare_results<float>(softmax_cpu, softmax_gpu, input_numel,
  //   input_dim_vec, "softmax");
  compare_results<float>(loss_cpu, loss_gpu, label_numel, label_dim_vec,
                         "loss");
  //   compare_results<float>(logit_grad_cpu, logit_grad_gpu, input_numel,
  //   input_dim_vec, "logit_grad");
}

}  // namespace operators
}  // namespace paddle