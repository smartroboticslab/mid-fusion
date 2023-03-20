cmake_minimum_required(VERSION 3.10)
project(ofusion++)

set(PROJECT_TEST_NAME segmentation)
set(UNIT_TEST_NAME ${PROJECT_TEST_NAME}-unittest)

file(GLOB PREPROCESSING "../../src/cpp/preprocessing/*.cpp")
file(GLOB SEGMENTATION "../../src/cpp/segmentation/*.cpp")

add_executable(${UNIT_TEST_NAME} segmentation_unittest.cpp ${PREPROCESSING} ${SEGMENTATION})
target_include_directories(${UNIT_TEST_NAME} PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(${UNIT_TEST_NAME} ${GTEST_BOTH_LIBRARIES} pthread
        ${OpenCV_LIBS} cnpy)
target_compile_definitions(${UNIT_TEST_NAME} PRIVATE FILEPATH="${CMAKE_CURRENT_LIST_DIR}")

gtest_add_tests(TARGET ${UNIT_TEST_NAME})
