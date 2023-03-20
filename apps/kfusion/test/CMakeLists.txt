cmake_minimum_required(VERSION 3.10)
project(kfusion_unit_testing)

# GTest Root - Change to reflect your install dir
#set(GTEST_ROOT ~/software/googletest/googletest)
find_package(GTest REQUIRED)
find_package( OpenCV REQUIRED )

set(TOON_INCLUDE_PATH ../TooN/install_dir/include)

set(OCTREELIB_ROOT ../../../) 
set(KFUSION_ROOT ../) 

enable_testing()
add_compile_options(-std=c++1y )
include_directories(${KFUSION_ROOT}/include ${KFUSION_ROOT}/kfusion/src 
  ${KFUSION_ROOT}/thirdparty ${OCTREELIB_ROOT}/utils ${OCTREELIB_ROOT}/lib
  ${TOON_INCLUDE_PATH} ${OpenCV_INCLUDE_DIRS})

add_subdirectory(preprocessing)
add_subdirectory(tracking)
add_subdirectory(segmentation)
