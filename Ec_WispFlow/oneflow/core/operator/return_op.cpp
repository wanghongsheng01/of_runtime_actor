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
#include "oneflow/core/operator/return_op.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

void ReturnOp::InitFromOpConf() {
  CHECK(op_conf().has_return_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_is_mutable(true);
}

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  *BlobDesc4BnInOp("out") = *BlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> ReturnOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> ReturnOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

REGISTER_OP(OperatorConf::kReturnConf, ReturnOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kReturnConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kReturnConf);

}  // namespace oneflow
