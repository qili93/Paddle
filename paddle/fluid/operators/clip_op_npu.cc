/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/clip_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;


void PrintTensor(const Tensor* tensor, const platform::Place &place, const std::string name) {
  std::cout << "=================== Print Tensor <" << name << ">, Place <" << tensor->place() << "> ===================" <<std::endl;
  framework::LoDTensor cpu_tensor;
  cpu_tensor.Resize(tensor->dims());
  framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);

  operators::TensorFormatter formatter;
  formatter.Print(cpu_tensor, name, "message");
}

template <typename DeviceContext, typename T>
class ClipNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    Tensor * min_tensor = ctx.HasInput("Min") ? const_cast<Tensor*>(ctx.Input<Tensor>("Min")) : nullptr;
    Tensor * max_tensor = ctx.HasInput("Max") ? const_cast<Tensor*>(ctx.Input<Tensor>("Max")) : nullptr;

    std::cout << "=================== ClipNPUKernel ===================" << std::endl;

    std::cout << "min_tensor = " << min_tensor << std::endl;
    std::cout << "max_tensor = " << max_tensor << std::endl;

    Tensor min_tensor_temp(x->type());
    Tensor max_tensor_temp(x->type());
    if (min_tensor == nullptr) {
      auto min_value = static_cast<T>(ctx.Attr<float>("min"));
      std::cout << "min_value = " << min_value << std::endl;
      min_tensor_temp.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&min_tensor_temp, min_value);
      min_tensor = &min_tensor_temp;
      PrintTensor(&min_tensor_temp, ctx.GetPlace(), "min_tensor_temp");
      PrintTensor(min_tensor, ctx.GetPlace(), "min_tensor");
    }

    if (max_tensor == nullptr) {
      auto max_value = static_cast<T>(ctx.Attr<float>("max"));
      std::cout << "max_value = " << max_value << std::endl;
      max_tensor_temp.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&max_tensor_temp, max_value);
      max_tensor = &max_tensor_temp;
      PrintTensor(&max_tensor_temp, ctx.GetPlace(), "max_tensor_temp");
      PrintTensor(max_tensor, ctx.GetPlace(), "max_tensor");
    }

    PrintTensor(min_tensor, ctx.GetPlace(), "min_tensor");
    PrintTensor(max_tensor, ctx.GetPlace(), "max_tensor");

    // const Tensor * min = nullptr;
    // const Tensor * max = nullptr;

    auto stream = ctx.template device_context<paddle::platform::NPUDeviceContext>().stream();

    // if (ctx.HasInput("Min")) {
    //   min = ctx.Input<Tensor>("Min");
    // } else {
    //   auto min_value = static_cast<T>(ctx.Attr<float>("max"));
    //   Tensor min_tensor(x->type());
    //   min_tensor.mutable_data<T>({1}, ctx.GetPlace());
    //   FillNpuTensorWithConstant<T>(&min_tensor, min_value);
    //   min = &min_tensor;
    // }

    // if (ctx.HasInput("Max")) {
    //   max = ctx.Input<Tensor>("Max");
    // } else {
    //   auto max_value = static_cast<T>(ctx.Attr<float>("max"));
    //   Tensor max_tensor(x->type());
    //   max_tensor.mutable_data<T>({1}, ctx.GetPlace());
    //   FillNpuTensorWithConstant<T>(&max_tensor, max_value);
    //   max = &max_tensor;
    // }

    const auto& runner = NpuOpRunner("ClipByValue", {*x, *min_tensor, *max_tensor}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ClipGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    std::cout << "=================== ClipGradNPUKernel ===================" << std::endl;

    auto * min_tensor = ctx.HasInput("Min") ? ctx.Input<Tensor>("Min") : nullptr;
    auto * max_tensor = ctx.HasInput("Max") ? ctx.Input<Tensor>("Max") : nullptr;


    auto min_val = ctx.Attr<float>("min");
    if (min_tensor) {
      Tensor min_data;
      framework::TensorCopy(*min_tensor, platform::CPUPlace(), ctx.template device_context<platform::DeviceContext>(), &min_data);
      ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
      min_val = static_cast<float>(min_data.data<T>()[0]);
    }

    auto max_val = ctx.Attr<float>("max");
    if (max_tensor) {
      Tensor max_data;
      framework::TensorCopy(*max_tensor, platform::CPUPlace(), ctx.template device_context<platform::DeviceContext>(), &max_data);
      ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
      max_val = static_cast<float>(max_data.data<T>()[0]);
    }

    LOG(INFO) << "min_val = " << min_val;
    LOG(INFO) << "max_val = " << max_val;

    auto stream = ctx.template device_context<paddle::platform::NPUDeviceContext>().stream();
    const auto& runner = NpuOpRunner("HardtanhGrad", {*x, *dout}, {*dx}, {{"min_val", min_val},{"max_val", max_val}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    clip, ops::ClipNPUKernel<plat::NPUDeviceContext, float>,
    ops::ClipNPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    clip_grad, ops::ClipGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::ClipGradNPUKernel<plat::NPUDeviceContext, plat::float16>);
