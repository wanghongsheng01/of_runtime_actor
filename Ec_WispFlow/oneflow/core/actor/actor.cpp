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
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/actor.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

void Actor::Init(const TaskProto& task_proto) {
  actor_id_ = task_proto.task_id();
  act_id_ = -1;
  // 构造Kernel
  for (const ExecNodeProto& node : task_proto.exec_sequence().exec_node()) {
    ExecKernel ek;
    ek.kernel = ConstructKernel(node.kernel_conf());
    exec_kernel_vec_.push_back(std::move(ek));
  }

  is_kernel_launch_synchronized_ =
      std::all_of(exec_kernel_vec_.cbegin(), exec_kernel_vec_.cend(),
                  [](const ExecKernel& ek) { return ek.kernel->IsKernelLaunchSynchronized(); });
  if (!is_kernel_launch_synchronized_) { CHECK_EQ(exec_kernel_vec_.size(), 1); }

  remaining_eord_cnt_ = 0;
  msg_handler_ = nullptr;
  eord_regst_desc_ids_.clear();

  // 创建输出Regst
  for (const auto& pair : task_proto.produced_regst_desc()) {
    Global<RegstMgr>::Get()->NewRegsts(pair.second, [this](Regst* regst) {
      produced_regsts_[regst->regst_desc_id()].emplace_back(regst);
    });
    int64_t regst_desc_id = pair.second.regst_desc_id();
    CHECK(name2regst_desc_id_.insert({pair.first, {regst_desc_id}}).second);
    produced_regst2expected_act_id_[regst_desc_id] = act_id_;
  }
  for (const auto& pair : produced_regsts_) {
    for (const auto& regst : pair.second) { produced_regst2reading_cnt_[regst.get()] = 0; }
  }

  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK(name2regst_desc_id_.find(pair.first) == name2regst_desc_id_.end());
    std::vector<int64_t>& regst_desc_id_vec = name2regst_desc_id_[pair.first];
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      regst_desc_id_vec.push_back(regst_desc_id);
      remaining_eord_cnt_ += 1;
    }
  }

  total_reading_cnt_ = 0;
  is_naive_consumed_eord_ = false;
  TakeOverNaiveConsumed(task_proto.consumed_regst_desc_id());
  TakeOverNaiveProduced(task_proto.produced_regst_desc());
  InitBnInOp2BlobInfo(task_proto);
  VirtualActorInit(task_proto);
}

// 构建输入RegstSlot
void Actor::TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
  auto res = GetNaiveOrCustomizedConsumedRegstDescName();
  bool is_naive_names = res.first == RegstNameType::kNaive;
  const HashSet<std::string>& names = res.second;

  for (const auto& pair : consumed_ids) {
    bool find_the_name = names.find(pair.first) != names.end();
    if (is_naive_names == find_the_name) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        naive_consumed_rs_.InsertRegstDescId(regst_desc_id);
      }
    }
  }
  naive_consumed_rs_.InitedDone();
}

// 构建输出RegstSlot
void Actor::TakeOverNaiveProduced(const PbMap<std::string, RegstDescProto>& produced_ids) {
  auto res = GetNaiveOrCustomizedProducedRegstDescName();
  bool is_naive_names = res.first == RegstNameType::kNaive;
  const HashSet<std::string>& names = res.second;

  for (const auto& pair : produced_ids) {
    bool find_the_name = names.find(pair.first) != names.end();
    if (is_naive_names == find_the_name) {
      naive_produced_rs_.InsertRegstDescId(pair.second.regst_desc_id());
    }
  }
  naive_produced_rs_.InitedDone();

  for (const auto& pair : produced_regsts_) {
    if (naive_produced_rs_.HasRegstDescId(pair.first) == false) { continue; }
    for (const auto& regst : pair.second) {
      CHECK_EQ(0, naive_produced_rs_.TryPushBackRegst(regst.get()));
    }
  }
}

// 根据task信息，构建ExecKernel中的bn_in_op2blob_info
void Actor::InitBnInOp2BlobInfo(const TaskProto& task_proto) {
  for (int64_t i = 0; i < exec_kernel_vec_.size(); ++i) {
    ExecKernel& ek = exec_kernel_vec_.at(i);
    const ExecNodeProto& node = task_proto.exec_sequence().exec_node().at(i);
    for (auto& pair : node.kernel_conf().op_attribute().arg_signature().bn_in_op2lbi()) {
      BlobInfo blob_info;
      blob_info.lbi = pair.second;
      const std::string& bn = pair.first;
      auto regst_desc_id_it = node.bn_in_op2regst_desc_id().find(bn);
      if (regst_desc_id_it != node.bn_in_op2regst_desc_id().end()
          && Global<RegstMgr>::Get()->HasRegstDescId(regst_desc_id_it->second)) {
        const int64_t regst_desc_id = regst_desc_id_it->second;
        blob_info.regst_desc_id = regst_desc_id;
        const RtRegstDesc& regst_desc =
            Global<RegstMgr>::Get()->RegstDesc4RegstDescId(regst_desc_id);
        blob_info.ordinal = regst_desc.GetOrdinalForLbi(blob_info.lbi);
        if (naive_produced_rs_.HasRegstDescId(regst_desc_id)) {
          blob_info.rs = &naive_produced_rs_;
        } else if (naive_consumed_rs_.HasRegstDescId(regst_desc_id)) {
          blob_info.rs = &naive_consumed_rs_;
        } else {
          blob_info.rs = nullptr;
        }
      } else {
        blob_info.regst_desc_id = -1;
        blob_info.ordinal = -1;
        blob_info.rs = nullptr;
      }
      ek.bn_in_op2blob_info.emplace(bn, std::move(blob_info));
    }
  }
}

// Actor正常运行过程中处理消息
int Actor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    remaining_eord_cnt_ -= 1;
    CHECK(eord_regst_desc_ids_.insert(msg.eord_regst_desc_id()).second);
    if (naive_consumed_rs_.HasRegstDescId(msg.eord_regst_desc_id())) {
      is_naive_consumed_eord_ = true;
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    // 如果是上游Actor发出的msg，表示输入Regst可读
    if (naive_consumed_rs_.HasRegstDescId(regst->regst_desc_id())) {
      CHECK_EQ(0, naive_consumed_rs_.TryPushBackRegst(regst));
      const auto& rdeq = naive_consumed_rs_.RegstDeq4RegstDescId(regst->regst_desc_id());
      CHECK(rdeq.empty() == false);
    // 如果是下游Actor发出的msg
    } else if (TryUpdtStateAsProducedRegst(regst) == 0) {
      // do nothing
    } else {
      NormalProcessCustomizedReadableRegstMsg(msg);
    }
    ActUntilFail();
  } else if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
    ActUntilFail();
  } else {
    UNIMPLEMENTED();
  }
  // handler halts
  bool has_naive_or_inplace = naive_consumed_rs_.total_regst_desc_cnt() != 0;
  bool naive_or_inplace_eord_and_empty = is_naive_consumed_eord_
      && naive_consumed_rs_.available_regst_desc_cnt() == 0;
  // 源Actor在满足结束条件时返回true
  bool customized_eord = IsCustomizedReadAlwaysUnReadyFromNow();
  if ((has_naive_or_inplace && naive_or_inplace_eord_and_empty)
      || (!has_naive_or_inplace && customized_eord)) {
    CHECK_EQ(naive_consumed_rs_.available_regst_desc_cnt(), 0);
    AsyncSendEORDMsgForAllProducedRegstDesc();
    if (remaining_eord_cnt_ == 0 && total_reading_cnt_ == 0) {
      OF_SET_MSG_HANDLER(nullptr);
      return 1;
    } else {
      OF_SET_MSG_HANDLER(&Actor::HandlerZombie);
      return 0;
    }
  }
  return 0;
}

// Actor有序退出过程中处理消息
int Actor::HandlerZombie(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    CHECK_GE(remaining_eord_cnt_, 1);
    remaining_eord_cnt_ -= 1;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) { UNIMPLEMENTED(); }
  } else {
    UNIMPLEMENTED();
  }
  if (remaining_eord_cnt_ == 0 && total_reading_cnt_ == 0) {
    msg_handler_ = nullptr;
    return 1;
  }
  return 0;
}

void Actor::TryLogActEvent(const std::function<void()>& DoAct) const {
  DoAct();
}

void Actor::ActUntilFail() {
  while (IsReadReady() && IsWriteReady()) {
    act_id_ += 1;
    TryLogActEvent([&] { Act(); });

    AsyncSendNaiveProducedRegstMsgToConsumer();
    AsyncSendNaiveConsumedRegstMsgToProducer();

    AsyncSendQueuedMsg();
  }
}

void Actor::AsyncSendNaiveProducedRegstMsgToConsumer() {
  VirtualAsyncSendNaiveProducedRegstMsgToConsumer();
}

void Actor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer();
}

void Actor::AsyncSendNaiveConsumedRegstMsgToProducer() {
  VirtualAsyncSendNaiveConsumedRegstMsgToProducer();
}

void Actor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  HandleConsumedNaiveDataRegstToProducer([](Regst* regst) { return true; });
}

// 向下游Actor发送RegstMsg，返回下游Actor个数
int64_t Actor::HandleRegstToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor) {
  auto regst_reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  CHECK_EQ(regst_reading_cnt_it->second, 0);
  regst->set_act_id(act_id_);

  int64_t real_consumer_cnt = 0;
  for (int64_t consumer : regst->consumers_actor_id()) {
    if (!IsAllowedActor(consumer)) { continue; }
    EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id_, consumer, regst));
    real_consumer_cnt += 1;
  }
  total_reading_cnt_ += real_consumer_cnt;
  regst_reading_cnt_it->second += real_consumer_cnt;
  return real_consumer_cnt;
}

bool Actor::IsReadReady() const {
  return naive_consumed_rs_.IsCurSlotReady();
}

bool Actor::IsWriteReady() const {
  return naive_produced_rs_.IsCurSlotReady();
}

// 启动Kernel序列
void Actor::AsyncLaunchKernel(std::function<Regst*(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    // 传入BnInOp2Blob函数，用来找到对应bn_in_op的Blob
    ek.kernel->Launch([&](const std::string& bn_in_op) -> Blob* {
      const auto blob_info_it = ek.bn_in_op2blob_info.find(bn_in_op);
      if (blob_info_it == ek.bn_in_op2blob_info.cend()) { return nullptr; }
      const BlobInfo& info = blob_info_it->second;
      if (info.regst_desc_id == -1) { return nullptr; }
      Regst* regst;
      if (info.rs != nullptr) {
        regst = info.rs->Front(info.regst_desc_id);
      } else {
        regst = Regst4RegstDescId(info.regst_desc_id);
      }
      if (regst == nullptr) { return nullptr; }
      if (info.ordinal >= 0) {
        return regst->GetBlobByOrdinal(info.ordinal);
      } else {
        return regst->GetBlobByLbi(info.lbi);
      }
    });
  }
}

void Actor::AsyncLaunchKernel() {
  AsyncLaunchKernel([](int64_t) -> Regst* {
    UNIMPLEMENTED();
    return nullptr;
  });
}

// 向下游Actor发送RegstMsg，从输出RegstSlot中移除对应的Regst
void Actor::HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                                   std::function<bool(int64_t)> IsAllowedActor) {
  tmp_regst_desc_id_vec_.clear();
  naive_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      if (RegstPreProcess(regst) == false) { return; }
      int64_t real_consumer_cnt = HandleRegstToConsumer(regst, IsAllowedActor);
      if (real_consumer_cnt > 0) { tmp_regst_desc_id_vec_.push_back(regst->regst_desc_id()); }
    }
  });
  naive_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void Actor::HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess) {
  HandleProducedNaiveDataRegstToConsumer(RegstPreProcess, [](int64_t) { return true; });
}

void Actor::HandleProducedNaiveDataRegstToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([](Regst*) { return true; });
}

// 向上游Actor发送RegstMsg，从输入RegstSlot中移除对应的Regst
void Actor::HandleConsumedNaiveDataRegstToProducer(std::function<bool(Regst*)> IsAllowedRegst) {
  tmp_regst_desc_id_vec_.clear();
  naive_consumed_rs_.ForEachFrontRegst([&](int64_t regst_desc_id, Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      if (IsAllowedRegst(regst) == false) { return; }
      // must access regst before sending it to producer
      tmp_regst_desc_id_vec_.push_back(regst->regst_desc_id());
      EnqueueAsyncMsg(
          ActorMsg::BuildRegstMsgToProducer(actor_id_, regst->producer_actor_id(), regst));
    }
  });
  naive_consumed_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

// 向下游Actor发送相关regst_desc的EordMsg
void Actor::AsyncSendEORDMsgForAllProducedRegstDesc() {
  for (auto& pair : produced_regsts_) {
    CHECK(!pair.second.empty());
    const RtRegstDesc* regst_desc = pair.second.front()->regst_desc();
    for (int64_t consumer : regst_desc->consumers_actor_id()) {
      Global<ActorMsgBus>::Get()->SendMsg(
          ActorMsg::BuildEordMsg(consumer, regst_desc->regst_desc_id()));
    }
  }
}

// 更新输出Regst的状态
int Actor::TryUpdtStateAsProducedRegst(Regst* regst) {
  auto reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  if (reading_cnt_it == produced_regst2reading_cnt_.end()) { return -1; }
  CHECK(produced_regsts_.find(regst->regst_desc_id()) != produced_regsts_.end());
  // 输出Regst需要被消费的次数减1
  CHECK_GE(reading_cnt_it->second, 1);
  reading_cnt_it->second -= 1;
  total_reading_cnt_ -= 1;
  // 如果输出Regst仍需被消费
  if (reading_cnt_it->second != 0) { return 0; }

  // 如果输出Regst已被下游Actor消费完成，则可写
  if (naive_produced_rs_.TryPushBackRegst(regst) != 0) {
    UpdtStateAsCustomizedProducedRegst(regst);
  }

  int64_t& expected_act_id = produced_regst2expected_act_id_[regst->regst_desc_id()];
  if (expected_act_id >= 0) {
    // 检查输出Regst的act_id与预期是否一致
    CHECK_EQ(regst->act_id(), expected_act_id);
  }
  // 更新expected_act_id
  // 与上面的检查语句一起，实质作用为，预言此Regst经过ActNumForEachOutput()次Act()后再次生产完成
  expected_act_id = regst->act_id() + ActNumForEachOutput(regst->regst_desc_id());
  return 0;
}

// 发送同步消息，或保存异步消息
void Actor::EnqueueAsyncMsg(const ActorMsg& msg) {
  if (is_kernel_launch_synchronized_
      && GetGlobalWorkStreamId()
             == Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(msg.dst_actor_id())) {
    Global<ActorMsgBus>::Get()->SendMsg(msg);
  } else {
    async_msg_queue_.push_back(msg);
  }
}

int64_t Actor::GetGlobalWorkStreamId() const {
  return Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(actor_id_);
}

// 把输出消息队列的内容通过ActorMsgBus发送出去
void Actor::AsyncSendQueuedMsg() {
  if (!async_msg_queue_.empty()) {
    std::deque<ActorMsg> msgs;
    msgs.swap(async_msg_queue_);
    for (const ActorMsg& msg : msgs) { Global<ActorMsgBus>::Get()->SendMsg(msg); }
  }
}

// 构建Actor
std::unique_ptr<Actor> NewActor(const TaskProto& task_proto) {
  const TaskType type = task_proto.task_type();
  Actor* rptr = NewObj<int32_t, Actor>(type);
  rptr->Init(task_proto);
  return std::unique_ptr<Actor>(rptr);
}

}  // namespace oneflow
