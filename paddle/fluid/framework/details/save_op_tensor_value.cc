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

#include "paddle/fluid/framework/details/save_op_tensor_value.h"
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace framework {
namespace details {

// This function only work when setting these two environment variables
// export FLAGS_save_tensor_value=1
// export GLOG_vmodule=layer=4
// Note: GLOG_vmodule=layer should be set to 4, or error will thrown
static void SaveTensorValue(const framework::LoDTensor& tensor,
                            const std::string& folder_path,
                            const std::string& var_name) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  std::string filename = var_name;
  std::size_t pos = filename.find("/");
  while (pos != std::string::npos) {
    filename.replace(pos, 1, ".");
    pos = filename.find("/");
  }

  std::string mkdir_cmd = "mkdir -p " + folder_path;
  PADDLE_ENFORCE_EQ(
      system(mkdir_cmd.c_str()), 0,
      platform::errors::NotFound("Cannot create folder %s", folder_path));

  std::string file_path = folder_path + filename + ".txt";
  std::ofstream fout(file_path);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout), true,
      platform::errors::NotFound("Cannot open %s to write", filename));

  operators::TensorFormatter formatter;
  fout << formatter.Format(tensor, filename, "");

  fout.close();
  VLOG(4) << "Save tensor to text file " << file_path;
}

static void SaveTensorCopyToHost(const std::string& op_type,
                                 const std::string& var_name,
                                 const framework::Tensor* tensor,
                                 const platform::Place& place,
                                 const std::string& folder_path) {
  framework::LoDTensor cpu_tensor;
  cpu_tensor.Resize(tensor->dims());
  framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
  SaveTensorValue(cpu_tensor, folder_path, var_name);
}

static void SaveTesnorValue(const std::string& op_type,
                            const framework::Scope& scope,
                            const std::string& tensor_name,
                            const std::string& var_name,
                            const platform::Place& place,
                            const std::string& value_type) {
  auto* var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::NotFound("In op=%s, can't find var:%s", op_type,
                                      var_name));

  if (!var->IsType<framework::LoDTensor>()) {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  framework::LoDTensor tensor = var->Get<framework::LoDTensor>();

  if (tensor.memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check OP(" << op_type << "), tensor:" << tensor_name
           << ", var_name:" << var_name << ", place:" << tensor.place()
           << ", numel:" << tensor.numel();

  std::string folder_path =
      "tensor_data/" + op_type + "/" + value_type + "/" + tensor_name + "/";

  if (platform::is_gpu_place(tensor.place())) {
    SaveTensorCopyToHost(op_type, var_name, &tensor, place, folder_path);
    return;
  }

  SaveTensorValue(tensor, folder_path, var_name);
}

void SaveOpTesnorValue(const framework::OperatorBase& op,
                       const framework::Scope& exec_scope,
                       const platform::Place& place) {
  LOG(INFO) << "Saving: " << op.DebugStringEx(&exec_scope);
  auto inputs = op.Inputs();
  auto outputs = op.Outputs();
  for (auto it = inputs.begin(); it != inputs.end(); ++it) {
    auto& input = *it;
    auto input_name = input.first;
    for (size_t i = 0; i < input.second.size(); ++i) {
      auto var_name = input.second[i];
      Variable* var = exec_scope.FindVar(var_name);
      if (var == nullptr) continue;
      SaveTesnorValue(op.Type(), exec_scope, input_name, var_name, place,
                      "InputVars");
    }
  }

  for (auto it = outputs.begin(); it != outputs.end(); ++it) {
    auto& output = *it;
    auto output_name = output.first;
    for (size_t i = 0; i < output.second.size(); ++i) {
      auto var_name = output.second[i];
      Variable* var = exec_scope.FindVar(var_name);
      if (var == nullptr) continue;
      SaveTesnorValue(op.Type(), exec_scope, output_name, var_name, place,
                      "OutputVars");
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
