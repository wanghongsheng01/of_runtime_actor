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
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

Shape::Shape(const std::initializer_list<int64_t>& dim_vec) : dim_vec_(dim_vec) { UpdateElemCnt(); }
Shape::Shape(const DimVector& dim_vec) : dim_vec_(dim_vec) { UpdateElemCnt(); }
Shape::Shape(DimVector&& dim_vec) : dim_vec_(std::move(dim_vec)) { UpdateElemCnt(); }

Shape::Shape(const ShapeProto& shape_proto) {
  dim_vec_.assign(shape_proto.dim().begin(), shape_proto.dim().end());
  UpdateElemCnt();
}

Shape& Shape::operator=(const Shape& shape) {
  dim_vec_ = shape.dim_vec_;
  UpdateElemCnt();
  return *this;
}

bool Shape::operator==(const Shape& rhs) const { return dim_vec_ == rhs.dim_vec_; }

std::string Shape::ToString() const {
  std::stringstream ss;
  int32_t idx = 0;
  ss << "(";
  for (int64_t dim : dim_vec_) {
    ss << dim;
    if (++idx != dim_vec_.size() || dim_vec_.size() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

std::string Shape::DebugStr() const { return ToString(); }

void Shape::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(dim_vec_.begin(), dim_vec_.end());
}

void Shape::Set(int64_t index, int64_t val) {
  dim_vec_.at(index) = val;
  UpdateElemCnt();
}

int64_t Shape::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= At(i); }
  return cnt;
}

int64_t Shape::Count(int64_t begin_axis) const { return Count(begin_axis, NumAxes()); }

void Shape::UpdateElemCnt() {
  elem_cnt_ = 1;
  for (int64_t s : dim_vec_) { elem_cnt_ *= s; }
  if (dim_vec_.size() == 0) { elem_cnt_ = 0; }
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.DebugStr();
  return out;
}


}  // namespace oneflow
