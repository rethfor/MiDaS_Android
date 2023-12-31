cmake_minimum_required(VERSION 3.22.1)
project("midaslite")

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../sdk/native/jni/abi-${ANDROID_ABI}/")

add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        native-lib.cpp)

find_package(OpenCV REQUIRED)

add_library(tensorflowlite SHARED IMPORTED)
set_target_properties(tensorflowlite PROPERTIES
                      IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../jniLibs/${ANDROID_ABI}/libtensorflowlite.so)

target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
        log
        ${OpenCV_LIBS}
        tensorflowlite
        )

include_directories("${OPENCV_DIR}/native/jni/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")