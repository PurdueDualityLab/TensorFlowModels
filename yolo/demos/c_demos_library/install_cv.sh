echo "OpenCV installation by learnOpenCV.com"
# Define OpenCV Version to install 
cvVersion="master"

# Clean build directories
rm -rf opencv/build
rm -rf opencv_contrib/build

# Create directory for installation
mkdir installation
mkdir installation/OpenCV-"$cvVersion"

# Save current working directory
cwd=$(pwd)
sudo apt -y update
sudo apt -y upgrade

sudo apt -y remove x264 libx264-dev

## Install dependencies
sudo apt -y install build-essential checkinstall cmake pkg-config yasm
sudo apt -y install git gfortran
sudo apt -y install libjpeg8-dev libpng-dev

sudo apt -y install software-properties-common
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt -y update

sudo apt -y install libjasper1
sudo apt -y install libtiff-dev

sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt -y install libxine2-dev libv4l-dev
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd "$cwd"

sudo apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt -y install libgtk2.0-dev libtbb-dev qt5-default
sudo apt -y install libatlas-base-dev
sudo apt -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt -y install libvorbis-dev libxvidcore-dev
sudo apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt -y install libavresample-dev
sudo apt -y install x264 v4l-utils

# Optional dependencies
sudo apt -y install libprotobuf-dev protobuf-compiler
sudo apt -y install libgoogle-glog-dev libgflags-dev
sudo apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout $cvVersion
cd ..

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout $cvVersion
cd ..



# cmake -D CMAKE_BUILD_TYPE=RELEASE \
#             -D CMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV-"$cvVersion" \
#             -D INSTALL_C_EXAMPLES=ON \
#             -D INSTALL_PYTHON_EXAMPLES=ON \
#             -D WITH_TBB=ON \
#             -D WITH_V4L=ON \
#             -D OPENCV_PYTHON3_INSTALL_PATH=$cwd/OpenCV-$cvVersion-py3/lib/python3.5/site-packages \
#         -D WITH_QT=ON \
#         -D WITH_OPENGL=ON \
#         -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
#         -D BUILD_EXAMPLES=ON ..

# make -j4
# make install

# cmake_minimum_required(VERSION 3.1)
# # Enable C++11
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# SET(OpenCV_DIR <OpenCV_Home_Dir>/installation/OpenCV-master/lib/cmake/opencv4)

# cd opencv
# mkdir build
# cd build