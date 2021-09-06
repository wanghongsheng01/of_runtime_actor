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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_helper.h"

namespace oneflow {

Kernel::~Kernel() {
}

void Kernel::InitBase(const KernelConf& kernel_conf) {
  kernel_conf_.reset(new KernelConf(kernel_conf));
}

void Kernel::Init(const KernelConf& kernel_conf) {
  InitBase(kernel_conf);
  VirtualKernelInit();
}

void Kernel::Launch(std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Forward(BnInOp2Blob);
}

const LogicalBlobId& Kernel::BnInOp2Lbi(const std::string& bn_in_op) const {
  return op_attribute().arg_signature().bn_in_op2lbi().at(bn_in_op);
}

void Kernel::Forward(std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (IsAllBlobEmpty(op_attribute().output_bns(), BnInOp2Blob) && IsStateless()) { return; }
  ForwardDataContent(BnInOp2Blob);
}

std::unique_ptr<const Kernel> ConstructKernel(const KernelConf& conf) {
  OperatorConf::OpTypeCase op_type = conf.op_attribute().op_conf().op_type_case();
  Kernel* rptr = kernel_registration::CreateKernel(conf);
  if (rptr == nullptr) { rptr = NewObj<OperatorConf::OpTypeCase, Kernel>(op_type, conf); }
  CHECK_NOTNULL(rptr);
  rptr->Init(conf);
  return std::unique_ptr<const Kernel>(rptr);
}

#define INSTANTIATE_KERNEL_IF(device_type) template class KernelIf<device_type>;

INSTANTIATE_KERNEL_IF(DeviceType::kCPU);

}  // namespace oneflow
