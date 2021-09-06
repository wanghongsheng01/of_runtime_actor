#include <iostream>
#include <fstream>
#include <vector>
#include <json/json.h>
#include <glog/logging.h>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/oneflow.h"

int main(int argc, char* argv[]) {
  const std::string file_path = "../test/job_set.prototxt";

  oneflow::JobSet job_set;
  if (!oneflow::TryParseProtoFromTextFile(file_path, &job_set)) {
    LOG(ERROR) << "prototxt parse failed: " << file_path;
    return -1;
  }

  oneflow::Global<oneflow::SessionGlobalObjectsScope>::New();
  JUST(oneflow::Global<oneflow::SessionGlobalObjectsScope>::Get()->Init());

  oneflow::Oneflow of{};
  of.Init(job_set);

  LOG(INFO) << "done";
  return 0;
}
