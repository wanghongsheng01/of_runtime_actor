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
#ifndef ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {

template<DeviceType device_type>
void Memcpy(void* dst, const void* src, size_t sz);

template<DeviceType device_type>
void Memset(void* dst, const char value, size_t sz);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
