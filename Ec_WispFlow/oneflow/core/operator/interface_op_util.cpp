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
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

void CheckShape(const Shape& shape) {
  FOR_RANGE(int, i, 1, shape.NumAxes()) { CHECK_GT(shape.At(i), 0); }
}

}  // namespace

Maybe<void> InterfaceOpUtil::InferOutBlobDesc(const InterfaceBlobConf& blob_conf,
                                              BlobDesc* out_blob_desc,
                                              const ParallelContext* parallel_ctx,
                                              const ParallelDesc& parallel_desc) {
  out_blob_desc->mut_shape() = Shape(blob_conf.shape());
  out_blob_desc->set_data_type(blob_conf.data_type());
  out_blob_desc->set_is_dynamic(blob_conf.is_dynamic());
  return Maybe<void>::Ok();
}

Maybe<void> InterfaceOpUtil::InferLogicalOutBlobDesc(const InterfaceBlobConf& blob_conf,
                                                     BlobDesc* out_blob_desc,
                                                     const ParallelDesc& parallel_desc) {
  CHECK_OR_RETURN(blob_conf.has_shape());
  out_blob_desc->mut_shape() = Shape(blob_conf.shape());
  CheckShape(out_blob_desc->shape());
  CHECK_GT(out_blob_desc->mut_shape().At(0), 0);
  CHECK_OR_RETURN(blob_conf.has_data_type());
  out_blob_desc->set_data_type(blob_conf.data_type());
  CHECK_OR_RETURN(blob_conf.has_is_dynamic());
  out_blob_desc->set_is_dynamic(blob_conf.is_dynamic());
  return Maybe<void>::Ok();
}

Maybe<void> InterfaceOpUtil::InitBlobConf(InterfaceBlobConf* blob_conf,
                                          const ParallelBlobConf& parallel_blob_conf) {
  BlobDesc blob_desc(parallel_blob_conf.logical_blob_desc_conf());
  blob_desc.shape().ToProto(blob_conf->mutable_shape());
  blob_conf->set_data_type(blob_desc.data_type());
  blob_conf->set_is_dynamic(blob_desc.is_dynamic());
  *blob_conf->mutable_parallel_distribution() = parallel_blob_conf.parallel_distribution();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
