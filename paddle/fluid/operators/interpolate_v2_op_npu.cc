/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/interpolate_v2_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

void PrintTensor(const Tensor* tensor, const std::string name) {
  std::cout << "=================== Print Tensor <" << name << ">, Place <" << tensor->place() << "> ===================" <<std::endl;
  framework::LoDTensor cpu_tensor;
  cpu_tensor.Resize(tensor->dims());
  framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);

  operators::TensorFormatter formatter;
  formatter.Print(cpu_tensor, name, "message");
}

template <typename DeviceContext, typename T>
class InterpolateV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto input_dims = input->dims();
    PADDLE_ENFORCE_EQ(input_dims.size(), 4UL,
        platform::errors::External("NPU Interpolate Kernel only support 4-D Tensor."));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);        
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input_dims, data_layout, &n, &c, &in_d, &in_h, &in_w);

    // LOG(INFO) << "input layout is " << input->layout();
    // LOG(INFO) << "output layout is " << output->layout();

    if (data_layout == DataLayout::kNHWC) {
      const_cast<Tensor*>(input)->set_layout(DataLayout::kNHWC);
      output->set_layout(DataLayout::kNHWC);
    }

    // LOG(INFO) << "input layout is " << input->layout();
    // LOG(INFO) << "output layout is " << output->layout();

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");
    // int align_mode = ctx.Attr<int>("align_mode");

    // To-do(qili93): need to support align_corners = true case, try ReSizeD
    PADDLE_ENFORCE_EQ(align_corners, false,
        platform::errors::InvalidArgument(
            "NPU Interpolate Kernel has diff when align_corners is true."));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;

    auto list_new_shape_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_shape_tensor.size() > 0) {
      // have size tensor
      // auto new_size = get_new_shape(list_new_shape_tensor);
      // out_h = new_size[0];
      // out_w = new_size[1];
      // PrintTensor(list_new_shape_tensor[0], "list_new_shape_tensor_0");
      // PrintTensor(list_new_shape_tensor[1], "list_new_shape_tensor_1");
      // LOG(INFO) << "list_new_shape_tensor.size() = " << list_new_shape_tensor.size();
      std::vector<int32_t> output_h(1);
      std::vector<int32_t> output_w(1);
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(ctx.GetPlace());
      framework::TensorToVector(*list_new_shape_tensor[0], *dev_ctx, &output_h);
      framework::TensorToVector(*list_new_shape_tensor[1], *dev_ctx, &output_w);
      out_h = output_h[0];
      out_w = output_w[0];
    } else {
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        if (scale_data.size() > 1) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else {
          scale_h = scale_data[0];
          scale_w = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      } else {
        if (scale.size() > 1) {
          scale_h = scale[0];
          scale_w = scale[1];

          PADDLE_ENFORCE_EQ(
              scale_w > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_w in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_w));
          PADDLE_ENFORCE_EQ(
              scale_h > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_h in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_h));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
      auto out_size = ctx.Input<Tensor>("OutSize");
      if (out_size != nullptr) {
        auto out_size_data = get_new_data_from_tensor<int>(out_size);
        out_h = out_size_data[0];
        out_w = out_size_data[1];
      }
    }
    PADDLE_ENFORCE_GT(out_h, 0, platform::errors::InvalidArgument(
                                    "out_h in Attr(out_shape) of Op(interpolate) "
                                    "should be greater than 0."));
    PADDLE_ENFORCE_GT(out_w, 0, platform::errors::InvalidArgument(
                                    "out_w in Attr(out_shape) of Op(interpolate) "
                                    "should be greater than 0."));
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {n, c, out_h, out_w};
    } else {
      dim_out = {n, out_h, out_w, c};
    }
    
    output->mutable_data<T>(dim_out, ctx.GetPlace());

    LOG(INFO) << "data_layout_str = " << data_layout_str;
    LOG(INFO) << "data_layout = " << data_layout;
    LOG(INFO) << "interp_method = " << interp_method;
    LOG(INFO) << "align_corners = " << align_corners;
    // LOG(INFO) << "align_mode = " << align_mode;
    LOG(INFO) << "n = " << n;
    LOG(INFO) << "c = " << c;
    LOG(INFO) << "in_d = " << in_d;
    LOG(INFO) << "in_h = " << in_h;
    LOG(INFO) << "in_w = " << in_w;
    LOG(INFO) << "out_h = " << out_h;
    LOG(INFO) << "out_w = " << out_w;
    LOG(INFO) << "scale_h = " << scale_h;
    LOG(INFO) << "scale_w = " << scale_w;

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*input, ctx.GetPlace(), output);
      return;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // std::string coordinate_transformation_mode = "half_pixel";
    // if (align_corners == true) {
    //   coordinate_transformation_mode = "align_corners";
    // }

    // std::string mode = "nearest";
    // if ("bilinear" == interp_method) {
    //   mode = "bilinear";
    // }
            
    NpuOpRunner runner;
    // runner.SetType("ResizeD");
    // runner.AddInput(*input)
    //       .AddOutput(*output)
    //       .AddAttr("sizes", std::vector<int32_t>{out_h, out_w})
    //       .AddAttr("coordinate_transformation_mode", coordinate_transformation_mode)
    //       .AddAttr("mode", mode);
    // runner.Run(stream);

    if ("bilinear" == interp_method) {
      runner.SetType("ResizeBilinearV2");
    } else if ("nearest" == interp_method) {
      runner.SetType("ResizeNearestNeighborV2");
    }
    runner.AddInput(*input)
          .AddInput(std::vector<int32_t>{out_h, out_w})
          .AddOutput(*output)
          .AddAttr("align_corners", align_corners)
          .AddAttr("half_pixel_centers", false);
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class InterpolateV2NPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");

    LOG(INFO) << "input tensor numel is: " << input->numel();
    LOG(INFO) << "input tensor type is: " << input->type();
    
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

    // LOG(INFO) << "input layout is " << input->layout();
    // LOG(INFO) << "output_grad layout is " << output_grad->layout();
    // LOG(INFO) << "input_grad layout is " << input_grad->layout();

    if (data_layout == DataLayout::kNHWC) {
      const_cast<Tensor*>(input)->set_layout(DataLayout::kNHWC);
      const_cast<Tensor*>(output_grad)->set_layout(DataLayout::kNHWC);
      input_grad->set_layout(DataLayout::kNHWC);
    }

    // LOG(INFO) << "input layout is " << input->layout();
    // LOG(INFO) << "output_grad layout is " << output_grad->layout();
    // LOG(INFO) << "input_grad layout is " << input_grad->layout();

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");
    // int align_mode = ctx.Attr<int>("align_mode");

    // To-do(qili93): need to support align_corners = true case, try ReSizeD
    PADDLE_ENFORCE_EQ(align_corners, false,
        platform::errors::InvalidArgument(
            "NPU Interpolate Kernel has diff when align_corners is true."));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;
    auto scale_tensor = ctx.Input<Tensor>("Scale");
    auto scale = ctx.Attr<std::vector<float>>("scale");
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      if (scale_data.size() > 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_w = scale_data[0];
        scale_h = scale_data[0];
      }
      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0, true,
          platform::errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    } else {
      if (scale.size() > 1) {
        scale_h = scale[0];
        scale_w = scale[1];
        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      }
    }
    if (scale_h > 0. && scale_w > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }
    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
    auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_size_tensor.size() > 0) {
      // have size tensor
      // auto new_size = get_new_shape(list_new_size_tensor);
      // out_h = new_size[0];
      // out_w = new_size[1];
      // PrintTensor(list_new_size_tensor[0], "list_new_shape_tensor_0");
      // PrintTensor(list_new_size_tensor[1], "list_new_shape_tensor_1");
      // LOG(INFO) << "list_new_shape_tensor.size() = " << list_new_size_tensor.size();
      std::vector<int32_t> output_h(1);
      std::vector<int32_t> output_w(1);
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(ctx.GetPlace());
      framework::TensorToVector(*list_new_size_tensor[0], *dev_ctx, &output_h);
      framework::TensorToVector(*list_new_size_tensor[1], *dev_ctx, &output_w);
      out_h = output_h[0];
      out_w = output_w[0];
    }

    framework::DDim dim_grad;
    if (data_layout == DataLayout::kNCHW) {
      dim_grad = {n, c, in_h, in_w};
    } else {
      dim_grad = {n, in_h, in_w, c};
    }
    input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());

    LOG(INFO) << "data_layout_str = " << data_layout_str;
    LOG(INFO) << "data_layout = " << data_layout;
    LOG(INFO) << "interp_method = " << interp_method;
    LOG(INFO) << "align_corners = " << align_corners;
    // LOG(INFO) << "align_mode = " << align_mode;
    LOG(INFO) << "n = " << n;
    LOG(INFO) << "c = " << c;
    LOG(INFO) << "in_d = " << in_d;
    LOG(INFO) << "in_h = " << in_h;
    LOG(INFO) << "in_w = " << in_w;
    LOG(INFO) << "out_h = " << out_h;
    LOG(INFO) << "out_w = " << out_w;
    LOG(INFO) << "scale_h = " << scale_h;
    LOG(INFO) << "scale_w = " << scale_w;
    
    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
      return;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // std::string coordinate_transformation_mode = "half_pixel";
    // if (align_corners == true) {
    //   coordinate_transformation_mode = "align_corners";
    // }

    // std::string mode = "nearest";
    // if ("bilinear" == interp_method) {
    //   mode = "bilinear";
    // }
            
    NpuOpRunner runner;
    // runner.SetType("ResizeGradD");
    // runner.AddInput(*output_grad)
    //       .AddOutput(*input_grad)
    //       .AddAttr("original_size", std::vector<int32_t>{in_h, in_w})
    //       .AddAttr("coordinate_transformation_mode", coordinate_transformation_mode)
    //       .AddAttr("mode", mode);
    // runner.Run(stream);

    // PrintTensor(input, "input");

    // LOG(INFO) << "output_grad tensor type is: " << output_grad->type();
    LOG(INFO) << "input tensor numel is: " << input->numel();
    LOG(INFO) << "input tensor type is: " << input->type();
    // LOG(INFO) << "input_grad tensor type is: " << input_grad->type();

    if ("bilinear" == interp_method) {
      runner.SetType("ResizeBilinearV2Grad")
            .AddInput(*output_grad)
            .AddInput(*input)
            .AddOutput(*input_grad)
            .AddAttr("align_corners", align_corners)
            .AddAttr("half_pixel_centers", false);
    } else if ("nearest" == interp_method) {
      runner.SetType("ResizeNearestNeighborV2Grad")
            .AddInput(*output_grad)
            .AddInput(std::vector<int32_t>{in_h, in_w})
            .AddOutput(*input_grad)
            .AddAttr("align_corners", align_corners)
            .AddAttr("half_pixel_centers", false);
    }
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    nearest_interp_v2, ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    nearest_interp_v2_grad, ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    bilinear_interp_v2, ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    bilinear_interp_v2_grad, ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, plat::float16>);
