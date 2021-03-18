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

USE_OP(batch_norm_grad);
// USE_OP_DEVICE_KERNEL(batch_norm_grad, CUDNN);

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
static void read_data(const platform::DeviceContext& ctx,
                      const framework::DDim dims,
                      framework::LoDTensor* tensor,
                      const std::string name,
                      const std::string filename) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  
  std::ifstream fin(filename.c_str());
  std::string data_str;
  for (std::string line; std::getline(fin, line);) {
    if (line.find("data:") == std::string::npos) continue;
    std::size_t start = line.find("[");
    std::size_t end = line.find("]", start);
    data_str = line.substr(start+1, end - start - 1);
  }
  fin.close();

  std::vector<T> data;
  std::istringstream iss(data_str);
  for (int i = 0; i < numel; ++i) {
    T value;
    iss >> value;
    data.push_back(value);
  }

  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
  //   print_data(data.data(), numel, framework::vectorize(dims), name);
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
const std::vector<int64_t> input_dim_vec = {1, 256, 56, 56};
const int64_t input_numel = std::accumulate(input_dim_vec.begin(), input_dim_vec.end(), 1, std::multiplies<int64_t>());
const std::vector<int64_t> scale_dim_vec = {256};
const int64_t scale_numel = std::accumulate(scale_dim_vec.begin(), scale_dim_vec.end(), 1, std::multiplies<int64_t>());
// attrs
const float epsilon = 1e-05;
const std::string data_layout = "NCHW";
const bool use_global_stats = false;
const bool is_test = false;

template <typename T>
void TestMain(const platform::DeviceContext& ctx,
              std::vector<float>& x_grad_out,
              std::vector<float>& scale_grad_out,
              std::vector<float>& bias_grad_out) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_bwd;

  framework::DDim input_dims = framework::make_ddim(input_dim_vec);
  framework::DDim scale_dims = framework::make_ddim(scale_dim_vec);

  // // --------------- backward ----------------------
  desc_bwd.SetType("batch_norm_grad");
  desc_bwd.SetInput(framework::GradVarName("Y"), {framework::GradVarName("Y")});
  desc_bwd.SetInput("Scale", {"Scale"});
  desc_bwd.SetInput("Bias", {"Bias"});
  desc_bwd.SetInput("SavedMean", {"SavedMean"});
  desc_bwd.SetInput("SavedVariance", {"SavedVariance"});
  desc_bwd.SetInput("X", {"X"});
  desc_bwd.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc_bwd.SetOutput(framework::GradVarName("Scale"), {framework::GradVarName("Scale")});
  desc_bwd.SetOutput(framework::GradVarName("Bias"), {framework::GradVarName("Bias")});
  desc_bwd.SetAttr("epsilon", epsilon);
  desc_bwd.SetAttr("data_layout", data_layout);
  desc_bwd.SetAttr("use_global_stats", use_global_stats);
  desc_bwd.SetAttr("is_test", is_test);
  desc_bwd.SetAttr("use_cudnn", true);
  desc_bwd.SetAttr("use_mkldnn", false);

  // other tensors
  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto save_mean_tensor = scope.Var("SavedMean")->GetMutable<framework::LoDTensor>();
  auto save_var_tensor = scope.Var("SavedVariance")->GetMutable<framework::LoDTensor>();
  // input tensors
  auto y_grad_tensor = scope.Var(framework::GradVarName("Y"))->GetMutable<framework::LoDTensor>();
  auto scale_tensor = scope.Var("Scale")->GetMutable<framework::LoDTensor>();
  auto bias_tensor = scope.Var("Bias")->GetMutable<framework::LoDTensor>();
  // output tensors
  auto x_grad_tensor = scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();
  auto scale_grad_tensor = scope.Var(framework::GradVarName("Scale"))->GetMutable<framework::LoDTensor>();
  auto bias_grad_tensor = scope.Var(framework::GradVarName("Bias"))->GetMutable<framework::LoDTensor>();

  // feed data to input tensors
  // feed_ones<T>(ctx, input_dims, x_tensor, "x_input");
  // feed_ones<T>(ctx, input_dims, y_grad_tensor, "y_grad");
  // feed_ones<T>(ctx, scale_dims, scale_tensor, "scale");
  // feed_ones<T>(ctx, scale_dims, bias_tensor, "bias");
  // feed_ones<T>(ctx, scale_dims, save_mean_tensor, "save_mean");
  // feed_ones<T>(ctx, scale_dims, save_var_tensor, "save_var");

  read_data<T>(ctx, input_dims, x_tensor, "x_input", "/workspace/bn_data/Input/X/dygraph_tmp_27.txt");
  read_data<T>(ctx, input_dims, y_grad_tensor, "y_grad", "/workspace/bn_data/Input/Y@GRAD/dygraph_tmp_28@GRAD.txt");
  read_data<T>(ctx, scale_dims, scale_tensor, "scale", "/workspace/bn_data/Input/Scale/bn2a_branch1_scale.txt");
  read_data<T>(ctx, scale_dims, bias_tensor, "bias", "/workspace/bn_data/Input/Bias/bn2a_branch1_offset.txt");
  read_data<T>(ctx, scale_dims, save_mean_tensor, "save_mean", "/workspace/bn_data/Input/SavedMean/dygraph_tmp_29.txt");
  read_data<T>(ctx, scale_dims, save_var_tensor, "save_var", "/workspace/bn_data/Input/SavedVariance/dygraph_tmp_30.txt");

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*x_grad_tensor, ctx, &x_grad_out);
  framework::TensorToVector(*scale_grad_tensor, ctx, &scale_grad_out);
  framework::TensorToVector(*x_grad_tensor, ctx, &bias_grad_out);
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
    // print_data(cpu_out.data(), numel, dims, name + "_cpu");
    // print_data(gpu_out.data(), numel, dims, name + "_gpu");
  } else {
    LOG(INFO) << "=========== Ouptut " << name << " is Equal in CPU and GPU ===========";
    // print_data(cpu_out.data(), numel, dims, name + "_cpu");
    // print_data(gpu_out.data(), numel, dims, name + "_gpu");
  }
}

TEST(test_batch_norm_op, compare_cpu_and_gpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> x_grad_out_cpu(input_numel);
  std::vector<float> scale_grad_out_cpu(scale_numel);
  std::vector<float> bias_grad_out_cpu(scale_numel);
  TestMain<float>(cpu_ctx, x_grad_out_cpu, scale_grad_out_cpu, bias_grad_out_cpu);

  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> x_grad_out_gpu(input_numel);
  std::vector<float> scale_grad_out_gpu(scale_numel);
  std::vector<float> bias_grad_out_gpu(scale_numel);
  TestMain<float>(gpu_ctx, x_grad_out_gpu, scale_grad_out_gpu, bias_grad_out_gpu);

  compare_results<float>(x_grad_out_cpu, x_grad_out_gpu, input_numel, input_dim_vec, "x_grad");
}

}  // namespace operators
}  // namespace paddle