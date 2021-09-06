#include <iostream>
#include <fstream>
#include <vector>
#include <json/json.h>
#include <glog/logging.h>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime.h"

int main(int argc, char* argv[]) {
  const std::string file_path = "../test/plan.prototxt";

  oneflow::Plan plan;
  if (!oneflow::TryParseProtoFromTextFile(file_path, &plan)) {
    LOG(ERROR) << "prototxt parse failed: " << file_path;
    return -1;
  }

  oneflow::Runtime rt(plan);

  LOG(INFO) << "done";
  return 0;
}
