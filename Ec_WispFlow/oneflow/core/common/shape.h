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
#ifndef ONEFLOW_CORE_COMMON_SHAPE_H_
#define ONEFLOW_CORE_COMMON_SHAPE_H_

#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow {

class Shape final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Shape);
  Shape() : elem_cnt_(0) {}
  explicit Shape(const DimVector& dim_vec);
  explicit Shape(DimVector&& dim_vec);
  explicit Shape(const ShapeProto& shape_proto);
  Shape(const std::initializer_list<int64_t>& dim_vec);
  ~Shape() = default;
  Shape& operator=(const Shape& shape);

  bool operator==(const Shape& rhs) const;
  bool operator!=(const Shape& rhs) const { return !(*this == rhs); }
  std::string DebugStr() const;
  std::string ToString() const;

  void ToProto(ShapeProto*) const;

  // Getters and Setters
  const DimVector& dim_vec() const { return dim_vec_; }
  int64_t elem_cnt() const { return elem_cnt_; }
  int64_t At(int64_t index) const { return dim_vec_.at(index); }
  void Set(int64_t index, int64_t val);
  int64_t NumAxes() const { return dim_vec_.size(); }
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t Count(int64_t begin_axis) const;

 private:
  void UpdateElemCnt();

  DimVector dim_vec_;
  int64_t elem_cnt_;
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::Shape> {
  size_t operator()(const oneflow::Shape& shape) const {
    size_t ret = 0;
    FOR_RANGE(int, i, 0, shape.NumAxes()) { ret ^= std::hash<int64_t>()(shape.At(i)); }
    return ret;
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_SHAPE_H_
