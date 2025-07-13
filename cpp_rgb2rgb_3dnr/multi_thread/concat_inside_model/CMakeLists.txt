# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(ENERZAiDemo CUDA CXX)

# TensorRT 설치 경로를 지정
set(TENSORRT_ROOT "/usr/src/tensorrt")

# TensorRT 헤더 파일 경로
include_directories(${TENSORRT_ROOT}/include)

# TensorRT 라이브러리 경로
link_directories(${TENSORRT_ROOT}/lib)

# tensorrt-config.cmake 파일이 있는 디렉토리를 지정
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "tensorrt-config.cmake")

# CUDA 찾기
find_package(OpenCV REQUIRED)

# CUDA 커널을 컴파일하기 위한 설정
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 파일 설정
set(CUDA_SOURCES cuda_functions.cu)

# CUDA 컴파일 옵션
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr)

find_package(CUDA REQUIRED)

# Find Boost
find_package(Boost COMPONENTS python numpy REQUIRED)

# Find Python
find_package(PythonLibs REQUIRED)

# 프로젝트 소스 파일 설정
set(SOURCES multi_thread.cpp ${CUDA_SOURCES} helper_functions.cpp)

# 실행 파일 생성
add_executable(ENERZAiDemo_multi_thread ${SOURCES})

# TensorRT 및 CUDA 라이브러리 링크
target_link_libraries(ENERZAiDemo_multi_thread PRIVATE nvinfer nvparsers nvinfer_plugin)
target_include_directories(ENERZAiDemo_multi_thread PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(ENERZAiDemo_multi_thread PRIVATE ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

# CUDA 컴파일러를 사용하는 경우
set_target_properties(ENERZAiDemo_multi_thread PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
