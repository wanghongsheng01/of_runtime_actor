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
#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

Blob::Blob(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* body_ptr) {
  mem_case_.reset(new MemoryCase{mem_case});
  blob_desc_ = blob_desc;
  dptr_ = body_ptr;
}

void Blob::CopyValidDataContentFrom(const Blob* rhs) {
  if (this == rhs) { return; }
  const size_t body_byte_size = ByteSizeOfBlobBody();
  CHECK_EQ(rhs->ByteSizeOfBlobBody(), body_byte_size);
  AutoMemcpy(mut_dptr(), rhs->dptr(), body_byte_size, mem_case(), rhs->mem_case());
}

}  // namespace oneflow
