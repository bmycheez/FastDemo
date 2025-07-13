# FindTensorRT.cmake

find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${CMAKE_PREFIX_PATH}/include)
find_library(TensorRT_LIBRARY NAMES nvinfer HINTS ${CMAKE_PREFIX_PATH}/lib)

set(TensorRT_FOUND FALSE)
if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
    set(TensorRT_FOUND TRUE)
endif()

if(TensorRT_FOUND)
    message(STATUS "Found TensorRT: ${TensorRT_INCLUDE_DIR}, ${TensorRT_LIBRARY}")
else()
    message(FATAL_ERROR "TensorRT not found. Make sure TensorRT is installed and the path is set correctly.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_LIBRARY)

