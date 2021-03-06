/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/kernel/return_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReturnKernel<device_type>::ForwardDataContent(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  LOG(INFO) << __func__;
  Blob* out = BnInOp2Blob("out");
  out->CopyValidDataContentFrom(BnInOp2Blob("in"));
  std::string result {"result is: "};
  for (int i = 0; i < out->shape().elem_cnt(); ++i) {
    result += std::to_string(out->dptr()[i]);
    result += " ";
  }
  LOG(INFO) << result;
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReturnConf, ReturnKernel);

}  // namespace oneflow
