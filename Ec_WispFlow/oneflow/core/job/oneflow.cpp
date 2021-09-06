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
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/sub_plan.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace {

Maybe<void> CompileCurJobOnMaster(Job* job, Plan* plan, bool need_job_complete) {
  const JobDesc& job_desc = GlobalJobDesc();
  double start = GetCurTime();
  Compiler().Compile(job, plan, need_job_complete);

  LOG(INFO) << "\njob_id: " << job_desc.job_id() << " , job_name: " << job_desc.job_name()
            << " , compile time: " << (GetCurTime() - start) / 1000000000.0 << " seconds.\n";
  return Maybe<void>::Ok();
}

void MergeSubPlanWithoutGenNetTopo(Plan* plan, std::vector<Plan>&& sub_plans) {
  CHECK(!sub_plans.empty());
  *plan = std::move(sub_plans.at(0));
  CHECK_EQ(sub_plans.size(), 1);
}

void AddJobName2JobId(const std::string& job_name, int64_t job_id) {
  CHECK(Global<JobName2JobId>::Get()->emplace(job_name, job_id).second);
}

REGISTER_FUNCTION_CONFIG_DEF().Bool("__is_user_function__", true, "is user defined function");

Maybe<void> CompileJobsAndMergePlans(const PbRpf<Job>& job_confs, Plan& plan) {
  std::vector<std::shared_ptr<Job>> jobs(job_confs.size());
  FOR_RANGE(int, i, 0, jobs.size()) { jobs.at(i).reset(new Job(job_confs.at(i))); }
  if (jobs.size() > 1) { UNIMPLEMENTED(); }

  std::vector<std::shared_ptr<Job>> function_jobs;
  function_jobs.reserve(jobs.size());
  FOR_RANGE(int, i, 0, jobs.size()) {
    JobDesc job_desc(jobs.at(i)->job_conf(), i);
    if (job_desc.Bool("__is_user_function__")) { function_jobs.push_back(jobs.at(i)); }
  }

  std::vector<Plan> sub_plans(jobs.size());
  FOR_RANGE(int64_t, i, 0, jobs.size()) {
    AddJobName2JobId(jobs.at(i)->job_conf().job_name(), i);
    auto scope = std::make_unique<GlobalJobDescScope>(jobs.at(i)->job_conf(), i);
    JUST(CompileCurJobOnMaster(jobs.at(i).get(), &sub_plans.at(i), true));
  }
  MergeSubPlanWithoutGenNetTopo(&plan, std::move(sub_plans));

  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> Oneflow::Init(const oneflow::JobSet& job_set) {
  CompileJobsAndMergePlans(job_set.job(), plan_);
  runtime_.reset(new Runtime(plan_));
  return Maybe<void>::Ok();
}

Oneflow::~Oneflow() {
}

}  // namespace oneflow
