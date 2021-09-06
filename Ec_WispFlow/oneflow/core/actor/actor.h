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
#ifndef ONEFLOW_CORE_ACTOR_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_H_

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/actor/register_slot.h"

namespace oneflow {

class Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor() = default;

  void Init(const TaskProto& task_proto);

  // 1: success, and actor finish
  // 0: success, and actor not finish
  int ProcessMsg(const ActorMsg& msg) { return (this->*msg_handler_)(msg); }

  int64_t thrd_id() const { return Global<IDMgr>::Get()->ThrdId4ActorId(actor_id_); }
  int64_t actor_id() const { return actor_id_; }

 protected:
  // Blob的LBI和内存相关信息
  struct BlobInfo {
    LogicalBlobId lbi;
    int64_t regst_desc_id;
    int64_t ordinal;
    RegstSlot* rs;
  };
  struct ExecKernel {
    std::unique_ptr<const Kernel> kernel;
    // 此Kernel中的BlobName与对应的BlobInfo
    HashMap<std::string, BlobInfo> bn_in_op2blob_info;
  };
  using MsgHandler = int (Actor::*)(const ActorMsg&);
  enum RegstNameType { kNaive = 0, kCustomized };

  // Util
  Actor() = default;
  virtual void VirtualActorInit(const TaskProto& task_proto) {}
  const std::vector<ExecKernel>& exec_kernel_vec() { return exec_kernel_vec_; }

  int64_t act_id() const { return act_id_; }

  // Msg Handler
  void set_msg_handler(MsgHandler val) { msg_handler_ = val; }
#define OF_SET_MSG_HANDLER(val)                                   \
  do {                                                            \
    LOG(INFO) << "actor " << actor_id() << " switch to " << #val; \
    set_msg_handler(static_cast<MsgHandler>(val));                \
  } while (0)

  // Common Handlers and related virtual method
  int HandlerNormal(const ActorMsg& msg);
  int HandlerZombie(const ActorMsg& msg);

  // Async Do on device_ctx_
  void AsyncLaunchKernel(std::function<Regst*(int64_t)> Regst4RegstDescId);
  void AsyncLaunchKernel();

  // Util For Derived Actor to Send Msg
  void EnqueueAsyncMsg(const ActorMsg&);
  void HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                              std::function<bool(int64_t)> IsAllowedActor);
  void HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess);
  void HandleProducedNaiveDataRegstToConsumer();

  void HandleConsumedNaiveDataRegstToProducer(std::function<bool(Regst*)> IsAllowedRegst);
  void AsyncSendEORDMsgForAllProducedRegstDesc();
  void AsyncSendQueuedMsg();

  // Get Regst
  int64_t HandleRegstToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor);

 protected:
  int64_t GetGlobalWorkStreamId() const;

  // Process Msg
  int TryUpdtStateAsProducedRegst(Regst* regst);

  // Act
  void ActUntilFail();
  virtual void Act() { UNIMPLEMENTED(); }
  virtual int64_t ActNumForEachOutput(int64_t regst_desc_id) const { return 1; }
  void TryLogActEvent(const std::function<void()>& Callback) const;

  // Ready
  bool IsReadReady() const;
  bool IsWriteReady() const;

  // Naive, Inplace Or Customized
  void TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids);
  void TakeOverNaiveProduced(const PbMap<std::string, RegstDescProto>& produced_ids);
  void InitBnInOp2BlobInfo(const TaskProto& task_proto);

  // Send Msgs
  void AsyncSendNaiveProducedRegstMsgToConsumer();
  virtual void VirtualAsyncSendNaiveProducedRegstMsgToConsumer();
  void AsyncSendNaiveConsumedRegstMsgToProducer();
  virtual void VirtualAsyncSendNaiveConsumedRegstMsgToProducer();

  // Customized Consumed virtual func
  virtual void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) { UNIMPLEMENTED(); }
  virtual bool IsCustomizedReadAlwaysUnReadyFromNow() const { return false; }
  virtual std::pair<RegstNameType, HashSet<std::string>>
  GetNaiveOrCustomizedConsumedRegstDescName() {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{});
  }

  // Customized Produced virtual func
  virtual void UpdtStateAsCustomizedProducedRegst(Regst* regst) { UNIMPLEMENTED(); }
  virtual std::pair<RegstNameType, HashSet<std::string>>
  GetNaiveOrCustomizedProducedRegstDescName() {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{});
  }

  int64_t actor_id_;
  // Act()执行过的次数
  int64_t act_id_;
  // 要执行的Kernel序列和相关Blob信息
  std::vector<ExecKernel> exec_kernel_vec_;
  // 保存所有输入输出RegstDesc的name和id
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;
  // 处理消息ActorMsg的函数，可能是 HandlerNormal 或 HandlerZombie
  MsgHandler msg_handler_;
  // 已收到结束消息kEordMsg的输入regst的集合
  HashSet<int64_t> eord_regst_desc_ids_;
  // 仍未收到结束消息kEordMsg的输入regst的个数
  int64_t remaining_eord_cnt_;

  // 此Actor的生产Regst即输出Regst，消费Regst即输入Regst
  // 输出regst_desc和对应的Regst实例
  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_regsts_;
  // 输出regst_desc和对应的预期act_id_
  HashMap<int64_t, int64_t> produced_regst2expected_act_id_;
  // 输出Regst和各自需要被消费的次数
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  // 输出Regst总共需要被消费的次数
  int64_t total_reading_cnt_;

  // 输出RegstSlot，保存可写的Regst
  RegstSlot naive_produced_rs_;
  // 输入RegstSlot，保存可读的Regst
  RegstSlot naive_consumed_rs_;
  // naive_produced_rs_中是否有regst收到了结束信号
  bool is_naive_consumed_eord_;

  // 输出的异步消息队列
  std::deque<ActorMsg> async_msg_queue_;
  bool is_kernel_launch_synchronized_;
  std::vector<int64_t> tmp_regst_desc_id_vec_;
};

std::unique_ptr<Actor> NewActor(const TaskProto& task_proto);

#define REGISTER_ACTOR(task_type, ActorType) REGISTER_CLASS(int32_t, task_type, Actor, ActorType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_H_
