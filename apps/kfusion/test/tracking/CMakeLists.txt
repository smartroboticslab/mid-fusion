cmake_minimum_required(VERSION 3.10)
project(octree_lib)
file(GLOB PREPROCESSING "../../src/cpp/preprocessing/*.cpp")
file(GLOB TRACKING "../../src/cpp/tracking/*.cpp")

#file(GLOB TRACKING "src/cpp/tracking/*.cpp")
add_library(lodepng SHARED ../../thirdparty/lodepng.cpp)

set(PROJECT_TEST_NAME tracking)
set(UNIT_TEST_NAME ${PROJECT_TEST_NAME}-jacobian-unittest)
add_executable(${UNIT_TEST_NAME} jacobians_unittest.cpp ${PREPROCESSING} ${TRACKING})
target_include_directories(${UNIT_TEST_NAME} PUBLIC ${GTEST_INCLUDE_DIRS}
        ${OpenCV_LIBS} )
target_link_libraries(${UNIT_TEST_NAME} ${GTEST_BOTH_LIBRARIES} ${OpenCV_LIBS} pthread lodepng)
target_compile_definitions(${UNIT_TEST_NAME} PRIVATE FILEPATH="${CMAKE_CURRENT_LIST_DIR}")

gtest_add_tests(TARGET ${UNIT_TEST_NAME})
