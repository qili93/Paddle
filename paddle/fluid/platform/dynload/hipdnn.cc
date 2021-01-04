/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/dynload/hipdnn.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace dynload {
std::once_flag hipdnn_dso_flag;
void* hipdnn_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

HIPDNN_DNN_ROUTINE_EACH(DEFINE_WRAP);
HIPDNN_DNN_ROUTINE_EACH_R2(DEFINE_WRAP);

#ifdef HIPDNN_DNN_ROUTINE_EACH_AFTER_R3
HIPDNN_DNN_ROUTINE_EACH_AFTER_R3(DEFINE_WRAP);
#endif

#ifdef HIPDNN_DNN_ROUTINE_EACH_AFTER_R3_LESS_R8
HIPDNN_DNN_ROUTINE_EACH_AFTER_R3_LESS_R8(DEFINE_WRAP);
#endif

#ifdef HIPDNN_DNN_ROUTINE_EACH_AFTER_R4
HIPDNN_DNN_ROUTINE_EACH_AFTER_R4(DEFINE_WRAP);
#endif

#ifdef HIPDNN_DNN_ROUTINE_EACH_R5
HIPDNN_DNN_ROUTINE_EACH_R5(DEFINE_WRAP);
#endif

#ifdef HIPDNN_DNN_ROUTINE_EACH_R6
HIPDNN_DNN_ROUTINE_EACH_R6(DEFINE_WRAP);
#endif

#ifdef HIPDNN_DNN_ROUTINE_EACH_R7
HIPDNN_DNN_ROUTINE_EACH_R7(DEFINE_WRAP);
#endif

#ifdef HIPDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7
HIPDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7(DEFINE_WRAP);
#endif

#ifdef HIPDNN_DNN_ROUTINE_EACH_AFTER_R7
HIPDNN_DNN_ROUTINE_EACH_AFTER_R7(DEFINE_WRAP);
#endif

#ifdef HIPDNN_DNN_ROUTINE_EACH_R8
HIPDNN_DNN_ROUTINE_EACH_R8(DEFINE_WRAP);
#endif

bool HasCUDNN() {
  std::call_once(hipdnn_dso_flag,
                 []() { hipdnn_dso_handle = GetCUDNNDsoHandle(); });
  return hipdnn_dso_handle != nullptr;
}

void EnforceHIPDNNLoaded(const char* fn_name) {
  PADDLE_ENFORCE_NOT_NULL(
      hipdnn_dso_handle,
      platform::errors::PreconditionNotMet(
          "Cannot load hipdnn shared library. Cannot invoke method %s.",
          fn_name));
}

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
