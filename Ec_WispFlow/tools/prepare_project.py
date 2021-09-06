import os
import argparse
import shutil
import subprocess

def prepare_jsoncpp(install_dir: str):
  os.chdir("jsoncpp")
  cmd = "git reset 1.9.4 --hard"
  subprocess.run(cmd, shell=True)

  path = "build"
  if not os.path.exists(path):
    os.mkdir(path)
  os.chdir(path)

  cmd = "cmake .. -DCMAKE_INSTALL_PREFIX=" + install_dir
  subprocess.run(cmd, shell=True)
  cmd = "make -j32 && make install"
  subprocess.run(cmd, shell=True)
  os.chdir("../..")
  print("jsoncpp done. current dir: ", os.getcwd())

def prepare_glog(install_dir: str):
  os.chdir("glog")
  cmd = "git reset v0.5.0 --hard"
  subprocess.run(cmd, shell=True)

  cmd = "cmake -S . -B build -G \"Unix Makefiles\" -DWITH_GFLAGS=OFF -DCMAKE_INSTALL_PREFIX=" + install_dir
  subprocess.run(cmd, shell=True)
  cmd = "cmake --build build"
  subprocess.run(cmd, shell=True)
  cmd = "cmake --build build --target test"
  subprocess.run(cmd, shell=True)
  cmd = "cmake --build build --target install"
  subprocess.run(cmd, shell=True)
  os.chdir("..")
  print("glog done. current dir: ", os.getcwd())

def prepare_protobuf(install_dir: str):
  os.chdir("protobuf")
  cmd = "git reset v3.17.3 --hard"
  subprocess.run(cmd, shell=True)
  cmd = "git submodule update --init --recursive"
  subprocess.run(cmd, shell=True)

  cmd = "./autogen.sh"
  subprocess.run(cmd, shell=True)
  cmd = "./configure --prefix=" + install_dir
  subprocess.run(cmd, shell=True)
  cmd = "make -j32 && make check -j32 && make install"
  subprocess.run(cmd, shell=True)
  os.chdir("..")
  print("protobuf done. current dir: ", os.getcwd())

def prepare_third_party(install_dir: str):
  path = "third_party"
  assert os.path.exists(path)
  os.chdir(path)
  path = "install"
  if not os.path.exists(path):
    os.mkdir(path)

  prepare_jsoncpp(install_dir)
  prepare_glog(install_dir)
  prepare_protobuf(install_dir)
  os.chdir("..")
  print("third_party done. current dir: ", os.getcwd())

def main():
  parser = argparse.ArgumentParser(description='Prepare for project')
  parser.add_argument("--project_path", "-p", type=str, default="./",
                      help='project path')
  parser.add_argument("--third_party_install_dir", "-t", type=str, default="~/Gb_usr/local/",
                      help='third party install dir')

  args = parser.parse_args()

  os.chdir(args.project_path)
  prepare_third_party(args.third_party_install_dir)

if __name__ == "__main__":
  main()
