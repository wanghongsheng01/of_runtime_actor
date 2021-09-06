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
#ifndef ONEFLOW_CORE_COMMON_MAYBE_H_
#define ONEFLOW_CORE_COMMON_MAYBE_H_

#include <memory>
#include <glog/logging.h>
#include "oneflow/core/common/type_traits.h"

namespace oneflow {

template<typename T, typename Enabled = void>
class Maybe;

template<typename T>
class Maybe<T, typename std::enable_if<!(std::is_same<T, void>::value || IsScalarType<T>::value)
                                       && !std::is_reference<T>::value>::type>
    final {
public:
  Maybe(const T& data) : data_(new T{data}) { }
  Maybe(const std::shared_ptr<T>& data) : data_(data) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  bool IsOk() { return true; }
  std::shared_ptr<T> Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return data_;
  }

private:
  std::shared_ptr<T> data_;
};

template<typename T>
class Maybe<T, typename std::enable_if<std::is_same<T, void>::value>::type> final {
public:
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  static Maybe Ok() { return Maybe(); }
  bool IsOk() { return true; }
  void Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {}

private:
  Maybe() = default;
};

template<typename T>
class Maybe<T, typename std::enable_if<IsScalarType<T>::value>::type> final {
public:
  Maybe(T data) : scalar_(data) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  void operator=(const Maybe& rhs) { scalar_ = rhs.scalar_; }

  bool IsOk() { return true; }
  T Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return scalar_;
  }

private:
  T scalar_ {T()};
};

#define __MaybeErrorStackCheckWrapper__(...) __VA_ARGS__

inline bool MaybeIsOk(Maybe<void>&& maybe) {
  return maybe.IsOk();
}

#define MAYBE_CONST_AUTO_REF const auto&

#define TRY(...) __MaybeErrorStackCheckWrapper__(__VA_ARGS__)
#define JUST(...)                                                              \
  ({                                                                           \
    MAYBE_CONST_AUTO_REF maybe = __MaybeErrorStackCheckWrapper__(__VA_ARGS__); \
    maybe;                                                                     \
  }).Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()
#define CHECK_JUST(...)                                                        \
  ([&]() {                                                \
    MAYBE_CONST_AUTO_REF maybe = __MaybeErrorStackCheckWrapper__(__VA_ARGS__); \
    return maybe;                                                              \
  })().Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#define CHECK_OK(...) CHECK(MaybeIsOk(std::move(__VA_ARGS__)))

}  // namespace oneflow

#define CHECK_OR_RETURN(expr)                                                    \
  if (!(expr))                                                                   \
    LOG(FATAL) << "Check failed: "

#define CHECK_EQ_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) == (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_GE_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) >= (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_GT_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) > (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_LE_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) <= (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_LT_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) < (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_NE_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) != (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_STREQ_OR_RETURN(lhs, rhs) CHECK_EQ_OR_RETURN(std::string(lhs), std::string(rhs))

#define CHECK_STRNE_OR_RETURN(lhs, rhs) CHECK_NE_OR_RETURN(std::string(lhs), std::string(rhs))

#define CHECK_NOTNULL_OR_RETURN(ptr) CHECK_OR_RETURN(ptr != nullptr)

#define CHECK_ISNULL_OR_RETURN(ptr) CHECK_OR_RETURN(ptr == nullptr)

#endif  // ONEFLOW_CORE_COMMON_MAYBE_H_
