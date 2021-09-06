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
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

DataType GetDataTypeFromBnInOpVec(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& bn_in_ops) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc) { return blob_desc->data_type(); }
  }
  return DataType::kInvalidDataType;
}

std::shared_ptr<Operator> CheckAndConstructOp(std::shared_ptr<const OperatorConf> op_conf) {
  Operator* rptr = NewObj<int32_t, Operator>(op_conf->op_type_case(), *op_conf);
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(op_conf->device_tag()));
  CHECK_EQ(device_type, DeviceType::kCPU);
  rptr->Init(op_conf);
  return std::shared_ptr<Operator>(rptr);
}

}  // namespace

Operator::Operator() : device_type_(DeviceType::kInvalidDevice) {}

void Operator::Init(const OperatorConf& op_conf) {
  Init(std::make_shared<const OperatorConf>(op_conf));
}

void Operator::Init(std::shared_ptr<const OperatorConf> op_conf) {
  op_conf_ = std::move(op_conf);
  device_type_ = CHECK_JUST(DeviceType4DeviceTag(op_conf_->device_tag()));
  InitFromOpConf();
  input_output_bns_.Reserve(input_bns().size() + output_bns().size());
  for (const auto& bn : input_bns()) { *input_output_bns_.Add() = bn; }
  for (const auto& bn : output_bns()) { *input_output_bns_.Add() = bn; }
}

const LogicalBlobId& Operator::BnInOp2Lbi(const std::string& bn_in_op) const {
  return arg_signature_.bn_in_op2lbi().at(bn_in_op);
}

const OperatorConf& Operator::op_conf() const {
  CHECK(op_conf_);
  return *op_conf_;
}

std::shared_ptr<const OperatorConf> Operator::shared_op_conf() const { return op_conf_; }

DeviceType Operator::device_type() const { return device_type_; }

const std::string& Operator::SoleIbn() const {
  CHECK_EQ(input_bns().size(), 1);
  return input_bns().Get(0);
}
const std::string& Operator::SoleObn() const {
  CHECK_EQ(output_bns().size(), 1);
  return output_bns().Get(0);
}
const std::string& Operator::SoleTbn() const {
  CHECK_EQ(tmp_bns().size(), 1);
  return tmp_bns().Get(0);
}

Maybe<const std::string*> Operator::obn4lbi(const LogicalBlobId& lbi) const {
  const auto& it = lbi2output_index_.find(lbi);
  CHECK_OR_RETURN(it != lbi2output_index_.end())
      << "no logical blob id found. lbn: " << lbi.op_name() << "/" << lbi.blob_name();
  return &output_bns().Get(it->second);
}

const PbRpf<std::string>& Operator::input_bns() const { return input_bns_; }

const PbRpf<std::string>& Operator::output_bns() const { return output_bns_; }

const PbRpf<std::string>& Operator::tmp_bns() const { return tmp_bns_; }

const PbRpf<std::string>& Operator::input_output_bns() const { return input_output_bns_; }

Maybe<void> Operator::FillOpParallelDesc(const ParallelDesc& parallel_desc) {
  return FillOpParallelDesc(std::make_shared<const ParallelDesc>(parallel_desc));
}

Maybe<void> Operator::FillOpParallelDesc(std::shared_ptr<const ParallelDesc> parallel_desc) {
  CHECK_OR_RETURN(!op_parallel_desc_);
  op_parallel_desc_ = std::move(parallel_desc);
  return Maybe<void>::Ok();
}

Maybe<const ParallelDesc> Operator::GetOpParallelDesc() const {
  CHECK_OR_RETURN(!op_parallel_desc_);
  return op_parallel_desc_;
}

namespace {

Maybe<void> FillLogicalBlobDesc(
    const std::function<Maybe<const BlobDesc>(int32_t)>& BlobDesc4Index,
    const PbRpf<std::string>& bns,
    std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>* index2logical_blob_desc_ptr) {
  CHECK_OR_RETURN(!(*index2logical_blob_desc_ptr));
  index2logical_blob_desc_ptr->reset(new std::vector<std::shared_ptr<const BlobDesc>>());
  (*index2logical_blob_desc_ptr)->reserve(bns.size());
  for (int32_t i = 0; i < bns.size(); ++i) {
    (*index2logical_blob_desc_ptr)->emplace_back(JUST(BlobDesc4Index(i)));
  }
  return Maybe<void>::Ok();
}

Maybe<void> FillLogicalBlobDesc(
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4BnInOp,
    const PbRpf<std::string>& bns,
    std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>* index2logical_blob_desc_ptr) {
  CHECK_OR_RETURN(!(*index2logical_blob_desc_ptr));
  index2logical_blob_desc_ptr->reset(new std::vector<std::shared_ptr<const BlobDesc>>());
  (*index2logical_blob_desc_ptr)->reserve(bns.size());
  for (const auto& bn : bns) {
    const BlobDesc& blob_desc = BlobDesc4BnInOp(bn);
    (*index2logical_blob_desc_ptr)->emplace_back(std::make_shared<const BlobDesc>(blob_desc));
  }
  return Maybe<void>::Ok();
}

Maybe<void> FillLogicalBlobDesc(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const PbRpf<std::string>& bns,
    std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>* index2logical_blob_desc_ptr) {
  JUST(FillLogicalBlobDesc(
      [&](const std::string& bn) -> const BlobDesc& {
        const BlobDesc* blob_desc = BlobDesc4BnInOp(bn);
        CHECK_NOTNULL(blob_desc);
        return *blob_desc;
      },
      bns, index2logical_blob_desc_ptr));
  return Maybe<void>::Ok();
}

Maybe<const BlobDesc> GetLogicalBlobDesc(
    const std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>& index2logical_blob_desc,
    int32_t index) {
  CHECK_OR_RETURN(index2logical_blob_desc);
  CHECK_LT_OR_RETURN(index, index2logical_blob_desc->size());
  return index2logical_blob_desc->at(index);
}

Maybe<void> FillLogicalBlobDescSignature(
    const PbRpf<std::string>& bns,
    const std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>& index2logical_blob_desc,
    PbMap<std::string, BlobDescProto>* bn_in_op2blob_desc) {
  CHECK_OR_RETURN(index2logical_blob_desc);
  CHECK_EQ_OR_RETURN(bns.size(), index2logical_blob_desc->size());
  for (int32_t i = 0; i < bns.size(); ++i) {
    index2logical_blob_desc->at(i)->ToProto(&(*bn_in_op2blob_desc)[bns.Get(i)]);
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> Operator::FillLogicalInBlobDesc(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  JUST(FillLogicalBlobDesc(BlobDesc4BnInOp, input_bns(), &input_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillLogicalInBlobDesc(
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4BnInOp) {
  JUST(FillLogicalBlobDesc(BlobDesc4BnInOp, input_bns(), &input_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillLogicalInBlobDesc(
    const std::function<Maybe<const BlobDesc>(int32_t)>& BlobDesc4InputIndex) {
  JUST(FillLogicalBlobDesc(BlobDesc4InputIndex, input_bns(), &input_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4Ibn(const std::string& ibn) const {
  return GetLogicalBlobDesc4InputIndex(JUST(GetInputIndex(ibn)));
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4InputIndex(int32_t index) const {
  return GetLogicalBlobDesc(input_index2logical_blob_desc_, index);
}

Maybe<void> Operator::FillLogicalOutBlobDesc(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  JUST(FillLogicalBlobDesc(BlobDesc4BnInOp, output_bns(), &output_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillLogicalOutBlobDesc(
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4BnInOp) {
  JUST(FillLogicalBlobDesc(BlobDesc4BnInOp, output_bns(), &output_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4Obn(const std::string& obn) const {
  return GetLogicalBlobDesc4OutputIndex(JUST(GetOutputIndex(obn)));
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4OutputIndex(int32_t index) const {
  return GetLogicalBlobDesc(output_index2logical_blob_desc_, index);
}

Maybe<const BlobDesc*> Operator::GetLogicalBlobDescPtr4OutputIndex(int32_t index) const {
  CHECK_OR_RETURN(output_index2logical_blob_desc_);
  CHECK_LT_OR_RETURN(index, output_index2logical_blob_desc_->size());
  CHECK_OR_RETURN(output_index2logical_blob_desc_->at(index));
  return output_index2logical_blob_desc_->at(index).get();
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4BnInOp(const std::string& bn) const {
  const auto& it = bn2index_pair_.find(bn);
  CHECK_OR_RETURN(it != bn2index_pair_.end());
  if (it->second.first == BlobNameTag::kInputBlobName) {
    return GetLogicalBlobDesc4InputIndex(it->second.second);
  } else if (it->second.first == BlobNameTag::kOutputBlobName) {
    return GetLogicalBlobDesc4OutputIndex(it->second.second);
  } else {
    UNIMPLEMENTED();
  }
}

Maybe<void> Operator::InferLogicalOutBlobDescsIf() {
  CHECK_OR_RETURN(input_index2logical_blob_desc_);
  CHECK_OR_RETURN(!output_index2logical_blob_desc_);
  std::vector<std::shared_ptr<BlobDesc>> output_logical_blob_desc_vec;
  output_logical_blob_desc_vec.resize(output_bns().size());
  for (auto& blob_desc : output_logical_blob_desc_vec) {
    blob_desc.reset(new BlobDesc(DataType::kInvalidDataType));
  }
  std::vector<std::shared_ptr<BlobDesc>> in_logical_blob_desc_vec;
  in_logical_blob_desc_vec.resize(input_bns().size());
  auto BlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
    const auto& it = bn2index_pair_.find(bn);
    CHECK(it != bn2index_pair_.end());
    if (it->second.first == BlobNameTag::kInputBlobName) {
      auto& ptr = in_logical_blob_desc_vec.at(it->second.second);
      if (!ptr) { ptr.reset(new BlobDesc(*input_index2logical_blob_desc_->at(it->second.second))); }
      return ptr.get();
    } else if (it->second.first == BlobNameTag::kOutputBlobName) {
      return output_logical_blob_desc_vec.at(it->second.second).get();
    } else {
      UNIMPLEMENTED();
    }
  };
  JUST(InferLogicalOutBlobDescs(BlobDesc4BnInOp, *JUST(GetOpParallelDesc())));
  output_index2logical_blob_desc_.reset(new std::vector<std::shared_ptr<const BlobDesc>>());
  output_index2logical_blob_desc_->resize(output_bns().size());
  for (int32_t i = 0; i < output_bns().size(); ++i) {
    output_index2logical_blob_desc_->at(i) = output_logical_blob_desc_vec.at(i);
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferBlobDescsIf(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  JUST(InferOutBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx));
  JUST(InferInternalBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx, job_desc));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferOutBlobDescsIf(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

Maybe<void> Operator::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  JUST(InferLogicalOutBlobDescs(GetBlobDesc4BnInOp, *JUST(GetOpParallelDesc())));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferInternalBlobDescsIf(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  return InferInternalBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, job_desc);
}

Maybe<void> Operator::InferInternalBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  return Maybe<void>::Ok();
}

namespace {

bool HasBlobDescWithField(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                          const PbRpf<std::string>& bn_in_ops,
                          std::function<bool(const BlobDesc*)> Predicator4BlobDesc) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc && Predicator4BlobDesc(blob_desc)) { return true; }
  }
  return false;
}

}  // namespace

void Operator::GenKernelConf(
    const std::function<const BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  for (const std::string& ibn : input_bns()) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(ibn);
    if (blob_desc == nullptr) { continue; }
  }

  CHECK_JUST(ToOpAttribute(kernel_conf->mutable_op_attribute()));
  if (HasBlobDescWithField(GetBlobDesc4BnInOp, output_bns(),
                           [](const BlobDesc* blob_desc) { return blob_desc->is_dynamic(); })) {
    kernel_conf->set_need_do_shape(true);
  }

  {
    DataType data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, output_bns());
    if (data_type == DataType::kInvalidDataType) {
      data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, input_bns());
    }
    kernel_conf->set_data_type(data_type);
  }

  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf);
}

void Operator::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {}

void Operator::AddLbi2OutputIndex(const LogicalBlobId& lbi, int32_t output_index) {
  CHECK(lbi2output_index_.emplace(lbi, output_index).second);
}

std::string Operator::Bn2ConfName(const std::string& bn) const {
  return GetStrValInPbFdOrPbRpf(GetCustomizedConf(), bn);
}

LogicalBlobId Operator::lbi4ibn(const std::string& input_bn) const {
  return GenLogicalBlobId(Bn2ConfName(input_bn));
}
LogicalBlobId Operator::lbi4obn(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  CHECK_EQ(output_bn, "out");
  ret.set_blob_name("out");
  return ret;
}

InputBlobModifier* Operator::EnrollInputBn(const std::string& ibn, bool has_diff) {
  LogicalBlobId lbi = lbi4ibn(ibn);
  auto* map = arg_modifier_signature_.mutable_ibn2input_blob_modifier();
  const auto& pair = map->insert({ibn, InputBlobModifier()});
  CHECK(pair.second);
  const int32_t input_index = input_bns_.size();
  CHECK(
      bn2index_pair_.emplace(ibn, std::make_pair(BlobNameTag::kInputBlobName, input_index)).second);
  *input_bns_.Add() = ibn;
  CHECK(mut_bn_in_op2lbi()->insert({ibn, lbi}).second);
  auto* ret = &pair.first->second;
  ret->set_requires_grad(has_diff);
  return ret;
}

const InputBlobModifier& Operator::InputBlobModifier4Ibn(const std::string& ibn) const {
  return arg_modifier_signature_.ibn2input_blob_modifier().at(ibn);
}

const OutputBlobModifier& Operator::OutputBlobModifier4Obn(const std::string& obn) const {
  return arg_modifier_signature_.obn2output_blob_modifier().at(obn);
}

InputBlobModifier* Operator::MutInputBlobModifier4Ibn(const std::string& ibn) {
  auto* map = arg_modifier_signature_.mutable_ibn2input_blob_modifier();
  return &map->at(ibn);
}

OutputBlobModifier* Operator::MutOutputBlobModifier4Obn(const std::string& obn) {
  auto* map = arg_modifier_signature_.mutable_obn2output_blob_modifier();
  return &map->at(obn);
}

OutputBlobModifier* Operator::EnrollOutputBn(const std::string& obn, bool has_diff) {
  LogicalBlobId lbi = lbi4obn(obn);
  auto* map = arg_modifier_signature_.mutable_obn2output_blob_modifier();
  const auto& pair = map->insert({obn, OutputBlobModifier()});
  CHECK(pair.second);
  auto* ret = &pair.first->second;
  const int32_t output_index = output_bns_.size();
  CHECK(bn2index_pair_.emplace(obn, std::make_pair(BlobNameTag::kOutputBlobName, output_index))
            .second);
  AddLbi2OutputIndex(lbi, output_index);
  *output_bns_.Add() = obn;
  CHECK(mut_bn_in_op2lbi()->insert({obn, lbi}).second);
  ret->set_requires_grad(has_diff);
  return ret;
}

std::string GenRepeatedBn(const std::string& bn_prefix, int32_t idx) {
  CHECK_GE(idx, 0);
  return bn_prefix + "_" + std::to_string(idx);
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf, DeviceType device_type) {
  std::shared_ptr<OperatorConf> dev_op_conf = std::make_shared<OperatorConf>(op_conf);
  dev_op_conf->set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type)));
  auto op = CheckAndConstructOp(dev_op_conf);
  return op;
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf) {
  return ConstructOp(op_conf, DeviceType::kCPU);
}

Maybe<int32_t> Operator::GetInputIndex(const std::string& ibn) const {
  auto it = bn2index_pair_.find(ibn);
  CHECK_OR_RETURN(it != bn2index_pair_.end());
  CHECK_EQ_OR_RETURN(it->second.first, BlobNameTag::kInputBlobName);
  return it->second.second;
}

Maybe<int32_t> Operator::GetOutputIndex(const std::string& obn) const {
  auto it = bn2index_pair_.find(obn);
  CHECK_OR_RETURN(it != bn2index_pair_.end());
  CHECK_EQ_OR_RETURN(it->second.first, BlobNameTag::kOutputBlobName);
  return it->second.second;
}

Maybe<int32_t> Operator::GetOutputIndex(const LogicalBlobId& lbi) const {
  auto it = lbi2output_index_.find(lbi);
  CHECK_OR_RETURN(it != lbi2output_index_.end());
  return it->second;
}

Maybe<void> Operator::ToOpAttribute(OpAttribute* op_attribute) const {
  *op_attribute->mutable_input_bns() = input_bns_;
  *op_attribute->mutable_output_bns() = output_bns_;
  *op_attribute->mutable_tmp_bns() = tmp_bns_;
  *op_attribute->mutable_op_conf() = op_conf();
  *op_attribute->mutable_arg_signature() = arg_signature_;
  if (input_index2logical_blob_desc_) {
    JUST(FillLogicalBlobDescSignature(
        input_bns(), input_index2logical_blob_desc_,
        op_attribute->mutable_logical_blob_desc_signature()->mutable_bn_in_op2blob_desc()));
  }
  if (output_index2logical_blob_desc_) {
    JUST(FillLogicalBlobDescSignature(
        output_bns(), output_index2logical_blob_desc_,
        op_attribute->mutable_logical_blob_desc_signature()->mutable_bn_in_op2blob_desc()));
  }
  return Maybe<void>::Ok();
}

LogicalBlobId GenLogicalBlobId(const std::string& lbn) {
  LogicalBlobId lbi;
  size_t pos = lbn.find('/');
  CHECK_NE(pos, std::string::npos) << "lbn: " << lbn;
  lbi.set_op_name(lbn.substr(0, pos));
  std::string blob_name_with_hit = lbn.substr(pos + 1);
  size_t vbar_pos = blob_name_with_hit.rfind('|');
  std::string blob_name_with_split_hit = blob_name_with_hit.substr(0, vbar_pos);
  size_t split_pos = blob_name_with_split_hit.rfind(':');
  lbi.set_blob_name(blob_name_with_split_hit.substr(0, split_pos));
  return lbi;
}

bool operator==(const OperatorConf& lhs, const OperatorConf& rhs) {
  return PbMd().Equals(lhs, rhs);
}

namespace {

Maybe<void> CheckOpInputSignature(const Operator& op, const OpNodeSignature& upstream_signature) {
  for (const auto& ibn : op.input_bns()) {
    {
      CHECK_OR_RETURN(upstream_signature.has_logical_blob_desc_signature());
      const auto& map = upstream_signature.logical_blob_desc_signature().bn_in_op2blob_desc();
      CHECK_OR_RETURN(map.find(ibn) != map.end());
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<Shape> GetLogicalShape(const Shape& physical_shape) {
  std::shared_ptr<Shape> logical_shape = std::make_shared<Shape>(physical_shape);
  return logical_shape;
}

}  // namespace oneflow
