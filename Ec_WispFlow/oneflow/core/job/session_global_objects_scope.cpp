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
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

SessionGlobalObjectsScope::SessionGlobalObjectsScope() {}

Maybe<void> SessionGlobalObjectsScope::Init() {
  Global<IDMgr>::New();
  Global<JobName2JobId>::New();
  return Maybe<void>::Ok();
}

SessionGlobalObjectsScope::~SessionGlobalObjectsScope() {
  Global<JobName2JobId>::Delete();
  Global<IDMgr>::Delete();
}

}  // namespace oneflow
