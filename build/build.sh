export Torch_DIR=../libtorch/share/cmake/Torch
export Caffe2_DIR=../libtorch/share/cmake/Caffe2
cmake -DCMAKE_PREFIX_PATH=../libtorch/ ..
make
