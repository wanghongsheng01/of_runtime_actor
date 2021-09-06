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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/input_op.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

void InputOp::InitFromOpConf() {
  CHECK(op_conf().has_input_conf());
  OutputBlobModifier* modifier = EnrollOutputBn("out", false);
  modifier->set_is_mutable(true);
  modifier->set_header_infered_before_compute(false);
}

Maybe<void> InputOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  BlobDesc* out_blob_desc = BlobDesc4BnInOp("out");
  JUST(InterfaceOpUtil::InferLogicalOutBlobDesc(op_conf().input_conf().blob_conf(), out_blob_desc,
                                                parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  JUST(InterfaceOpUtil::InferOutBlobDesc(op_conf().input_conf().blob_conf(), out_blob_desc,
                                         parallel_ctx, *JUST(GetOpParallelDesc())));
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kInputConf, InputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kInputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kInputConf);

}  // namespace oneflow
