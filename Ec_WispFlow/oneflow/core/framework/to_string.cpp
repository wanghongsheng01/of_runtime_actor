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
#include <map>
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/device_registry_manager.h"

namespace oneflow {

Maybe<std::string> DeviceTag4DeviceType(DeviceType device_type) {
  auto device_type_to_tag = DeviceRegistryMgr::Get().DeviceType4Tag();
  auto it = device_type_to_tag.find(device_type);
  if (it == device_type_to_tag.end()) {
    UNIMPLEMENTED();
    return std::string{""};
  } else {
    return it->second;
  }
}

Maybe<DeviceType> DeviceType4DeviceTag(const std::string& device_tag) {
  auto device_tag_to_type = DeviceRegistryMgr::Get().DeviceTag4Type();
  if (device_tag_to_type.find(device_tag) == device_tag_to_type.end()) {
    UNIMPLEMENTED();
  }
  return device_tag_to_type[device_tag];
}

}  // namespace oneflow
