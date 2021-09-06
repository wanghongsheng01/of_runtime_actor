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
#ifndef ONEFLOW_CORE_COMMON_JSON_UTIL_H_
#define ONEFLOW_CORE_COMMON_JSON_UTIL_H_

#include <json/json.h>
#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
class BuildContainerFromJson final {
public:
  static void build_set(const Json::Value& array, std::unordered_set<T>& res) {
    res.clear();
    for (const Json::Value& id : array) {
      res.emplace(T{id});
    }
  }

  static void build_vector(const Json::Value& array, std::vector<T>& res) {
    res.clear();
    for (const Json::Value& id : array) {
      res.emplace_back(T{id});
    }
  }
};

template<>
class BuildContainerFromJson<int64_t> final {
public:
  static void build_set(const Json::Value& array, std::unordered_set<int64_t>& res) {
    res.clear();
    for (const Json::Value& id : array) {
      res.emplace(id.asInt64());
    }
  }

  static void build_vector(const Json::Value& array, std::vector<int64_t>& res) {
    res.clear();
    for (const Json::Value& id : array) {
      res.emplace_back(id.asInt64());
    }
  }
};

template<>
class BuildContainerFromJson<std::string> final {
public:
  static void build_vector(const Json::Value& array, std::vector<std::string>& res) {
    res.clear();
    for (const Json::Value& id : array) {
      res.emplace_back(id.asString());
    }
  }

  template<typename V>
  static void build_hashmap(const Json::Value& dict, HashMap<std::string, V>& res) {
    res.clear();
    for (const std::string& name : dict.getMemberNames()) {
      res.emplace(std::pair<
        std::string, V>{name, V{dict[name]}});
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_JSON_UTIL_H_
