# tensorrt-config.cmake

# Exported from TensorRT
# Provides the include directories and library to use TensorRT in a CMake project.

# The version of TensorRT
set(TensorRT_VERSION_MAJOR 8)
set(TensorRT_VERSION_MINOR 0)

# Define the include directories
set(TensorRT_INCLUDE_DIRS "${CMAKE_PREFIX_PATH}/include")

# Define the library directories
set(TensorRT_LIBRARY_DIRS "${CMAKE_PREFIX_PATH}/lib")

# Define the libraries to link against
set(TensorRT_LIBRARIES nvinfer)

# Provide the version information
set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}")

# Add the include directories to the user's build
list(APPEND TensorRT_INCLUDE_DIRS "${CMAKE_PREFIX_PATH}/include")

# Add the library directories to the user's build
list(APPEND TensorRT_LIBRARY_DIRS "${CMAKE_PREFIX_PATH}/lib")

# Provide the user with variables
set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIRS} CACHE STRING "TensorRT include directories" FORCE)
set(TensorRT_LIBRARY_DIRS ${TensorRT_LIBRARY_DIRS} CACHE STRING "TensorRT library directories" FORCE)
set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES} CACHE STRING "TensorRT libraries" FORCE)
set(TensorRT_VERSION ${TensorRT_VERSION} CACHE STRING "TensorRT version" FORCE)

