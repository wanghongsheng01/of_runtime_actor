# Minimum CMake required
cmake_minimum_required(VERSION 3.14.0)

project(WispFlow)

set(CMAKE_BUILD_TYPE "Debug")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(EXTRA_CXX_FLAGS "-std=c++11 -Wall -Wno-sign-compare -Wno-unused-function -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_CXX_FLAGS} -g -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
message("-- CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})

set(oneflow_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

# Modules
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir}/third_party)
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir})
include(proto2cpp)

include(third_party)

include_directories(~/Gb_usr/local/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories(~/Gb_usr/local/lib)

include_directories(${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE oneflow_all_src "${PROJECT_SOURCE_DIR}/oneflow/core/*.*")

foreach(oneflow_single_file ${oneflow_all_src})
  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.proto$")
    list(APPEND of_all_proto ${oneflow_single_file})
    set(group_this ON)
  endif()
endforeach()

# proto files
foreach(proto_name ${of_all_proto})
  file(RELATIVE_PATH proto_rel_name ${PROJECT_SOURCE_DIR} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${PROJECT_SOURCE_DIR}
                               ${of_all_rel_protos})

add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib) 
add_library(of_ccobj SHARED ${oneflow_all_src})
target_link_libraries(of_ccobj of_protoobj protobuf jsoncpp glog)

add_executable(main test/test_job.cpp)
target_link_libraries(main of_ccobj of_protoobj protobuf jsoncpp glog)
