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
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_H_

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/kernel/kernel_registration.h"
#include "oneflow/core/operator/op_conf_util.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel();

  void Init(const KernelConf&);

  void Launch(std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  const LogicalBlobId& BnInOp2Lbi(const std::string& bn_in_op) const;
  const OperatorConf& op_conf() const { return op_attribute().op_conf(); }
  const OpAttribute& op_attribute() const { return kernel_conf().op_attribute(); }
  /*
   * return true means all below must be guaranteed when `Launch` function return:
   * 1) all out blob header has been set (e.g. SyncSetHeadKernel)
   * 2) all asynchronous task has been queued (e.g. NCCL related kernel)
   */
  virtual bool IsKernelLaunchSynchronized() const { return true; }

  virtual void Forward(std::function<Blob*(const std::string&)> BnInOp2Blob) const;

 protected:
  Kernel() {}
  void InitBase(const KernelConf&);
  virtual void VirtualKernelInit() {}
  const KernelConf& kernel_conf() const {
    CHECK_NOTNULL(kernel_conf_.get());
    return *kernel_conf_;
  }

  // TODO(niuchong) : rename ForwardDataContent to ForwardBody
  virtual void ForwardDataContent(std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual bool IsStateless() const { return false; }

 private:
  std::unique_ptr<KernelConf> kernel_conf_;
};

template<DeviceType device_type>
class KernelIf : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelIf);
  virtual ~KernelIf() = default;

 protected:
  KernelIf() = default;

  void CopyField(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const Blob* from_blob, const std::vector<std::string>& to_bns,
                 void (Blob::*Copy)(const Blob*)) const {
    for (const std::string& to_bn : to_bns) { (BnInOp2Blob(to_bn)->*Copy)(from_blob); }
  }
  void CopyField(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const std::vector<std::string>& from_bns, const std::vector<std::string>& to_bns,
                 void (Blob::*Copy)(const Blob*)) const {
    if (from_bns.size() == 1) {
      const Blob* in_blob = BnInOp2Blob(from_bns[0]);
      CopyField(BnInOp2Blob, in_blob, to_bns, Copy);
    } else if (to_bns.size() == 1) {
      Blob* in_blob = BnInOp2Blob(from_bns[0]);
      Blob* out_blob = BnInOp2Blob(to_bns[0]);
      (out_blob->*Copy)(in_blob);
    } else {
      CHECK_EQ(from_bns.size(), to_bns.size());
      FOR_RANGE(size_t, i, 0, from_bns.size()) {
        Blob* in_blob = BnInOp2Blob(from_bns[i]);
        Blob* out_blob = BnInOp2Blob(to_bns[i]);
        (out_blob->*Copy)(in_blob);
      }
    }
  }
};

#define REGISTER_KERNEL_CREATOR(k, f) \
  REGISTER_CLASS_CREATOR(OperatorConf::OpTypeCase, k, Kernel, f, const KernelConf&)

std::unique_ptr<const Kernel> ConstructKernel(const KernelConf&);

}  // namespace oneflow

#define MAKE_DEVICE_TYPE_KERNEL_CREATOR_ENTRY(kernel_class, device_type) \
  {device_type, []() { return new kernel_class<device_type>(); }},

#define ADD_DEVICE_TYPE_KERNEL_CREATOR(op_type_case, kernel_class)                              \
  namespace {                                                                                   \
                                                                                                \
  Kernel* OF_PP_CAT(CreateKernel, __LINE__)(const KernelConf& kernel_conf) {                    \
    static const HashMap<int, std::function<Kernel*()>> creators = {                            \
        MAKE_DEVICE_TYPE_KERNEL_CREATOR_ENTRY(kernel_class, DeviceType::kCPU)};                 \
    DeviceType device_type =                                                                    \
        CHECK_JUST(DeviceType4DeviceTag(kernel_conf.op_attribute().op_conf().device_tag()));    \
    return creators.at(device_type)();                                                          \
  }                                                                                             \
                                                                                                \
  REGISTER_KERNEL_CREATOR(op_type_case, OF_PP_CAT(CreateKernel, __LINE__));                     \
  }

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_H_
