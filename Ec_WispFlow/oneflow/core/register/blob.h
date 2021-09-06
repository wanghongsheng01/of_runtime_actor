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
#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

class Blob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* body_ptr);
  ~Blob() = default;

  const MemoryCase& mem_case() const {
    CHECK_NOTNULL(mem_case_);
    return *mem_case_;
  }
  const BlobDesc& blob_desc() const {
    CHECK_NOTNULL(blob_desc_);
    return *blob_desc_;
  }
  const float* dptr() const {
    CHECK_NOTNULL(blob_desc_);
    CHECK_EQ(blob_desc_->data_type(), DataType::kFloat);
    return static_cast<const float*>(dptr_);
  }
  float* mut_dptr() {
    CHECK_NOTNULL(blob_desc_);
    CHECK_EQ(blob_desc_->data_type(), DataType::kFloat);
    return static_cast<float*>(dptr_);
  }
  const Shape& shape() const { return blob_desc_->shape(); }
  size_t ByteSizeOfBlobBody() const { return blob_desc_->ByteSizeOfBlobBody(); }

  void CopyValidDataContentFrom(const Blob* rhs);
  bool IsBodyEmpty() { return shape().elem_cnt() == 0; }

 private:
  std::unique_ptr<MemoryCase> mem_case_;
  const BlobDesc* blob_desc_ {nullptr};
  void* dptr_ {nullptr};
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
