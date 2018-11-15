export Torch_DIR=../libtorch/share/cmake/Torch
export Caffe2_DIR=../libtorch/share/cmake/Caffe2
cmake -DCMAKE_PREFIX_PATH=../libtorch/ ..
#cmake -DCMAKE_PREFIX_PATH=../libtorch/ -DCMAKE_BUILD_TYPE=release -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DARCHIVE_INSTALL_DIR=. -G "Unix Makefiles" ..
make
