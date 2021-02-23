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
static void print_tensor_data(const paddle::platform::DeviceContext& ctx,
                             paddle::framework::LoDTensor* tensor,
                             const char * name) {
  size_t numel = static_cast<size_t>(paddle::framework::product(tensor->dims()));
  std::vector<T> data(numel);
  paddle::framework::TensorToVector(*tensor, ctx, &data);

  printf("=============%s============\n", name);
  size_t stride = tensor->dims()[1];
  size_t index = 0;
  while(index < numel) {
    printf("%2.1f ", data[index]);
    if((index+1) % stride == 0) printf("\n");
    index ++;
  }
}

const int input_height = 32;
const int input_width = 64;
const int input_numel = input_height * input_width;
const float dropout_prob = 0.0;
const std::string dropout_implementation = "downgrade_in_infer";
const bool fix_seed = false;
const int seed = 0;

template <typename T>
void TestDropOut(const platform::DeviceContext& ctx, const bool use_cudnn = false) {
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
  print_tensor_data<T>(ctx, output_tensor, "output");
}

TEST(test_dropout_op, cpu_place) {
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  TestDropOut<float>(ctx, false);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_dropout_op, gpu_place) {
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);
  TestDropOut<float>(ctx, false);
}
#endif

}  // namespace operators
}  // namespace paddle
