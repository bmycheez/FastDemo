#include "helper_functions.h"

// 헬퍼 함수: 엔진 로드
nvinfer1::ICudaEngine* loadEngine(const std::string& enginePath, nvinfer1::ILogger* logger) {
    // TensorRT 런타임 초기화
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(*logger);

    // 엔진 로드
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile.is_open()) {
        logger->log(nvinfer1::ILogger::Severity::kERROR, ("Error opening engine file at path: " + enginePath).c_str());
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::unique_ptr<char[]> engineData(new char[fileSize]);
    engineFile.read(engineData.get(), fileSize);
    engineFile.close();

    // TensorRT 엔진 생성
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.get(), fileSize, nullptr);

    return engine;
}

// 헬퍼 함수: gstreamer pipeline text
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// 헬퍼 함수: Dims 정보를 문자열로 변환
std::string dimsToString(const nvinfer1::Dims& dims) {
    std::string result = "(";
    for (int i = 0; i < dims.nbDims; ++i) {
        result += std::to_string(dims.d[i]);
        if (i < dims.nbDims - 1) {
            result += ", ";
        }
    }
    result += ")";
    return result;
}

// 헬퍼 함수: Dims 정보를 이용하여 Tensor 크기 계산
size_t volume(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

// 헬퍼 함수: 텐서 정보 출력
void printTensorInfo(nvinfer1::ICudaEngine* engine) {
    std::cout << "Tensor Information:" << std::endl;

    for (int i = 0; i < engine->getNbBindings(); ++i) {
        std::cout << "Tensor Name: " << engine->getBindingName(i) << std::endl;
        std::cout << "Tensor Dimension: " << dimsToString(engine->getBindingDimensions(i)) << std::endl;
        std::cout << "Tensor Type: ";
        if (engine->getBindingDataType(i) == nvinfer1::DataType::kFLOAT) {
            std::cout << "float";
        } else {
            std::cout << "Unknown";
        }
        std::cout << std::endl;
        std::cout << "Tensor Size: " << volume(engine->getBindingDimensions(i)) << std::endl;
        std::cout << "----------------------------------" << std::endl;
    }
}

// 헬퍼 함수: 열거형을 기반으로 텐서의 데이터 타입 크기를 계산하는 사용자 정의 함수
size_t getDataTypeSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return sizeof(float);
        case nvinfer1::DataType::kHALF: return sizeof(uint16_t);
        case nvinfer1::DataType::kINT8: return sizeof(int8_t);
        // 추가 데이터 타입에 대한 처리 필요
        default: return 0; // 처리되지 않은 데이터 타입에 대한 기본값
    }
}

// allocateCpuGpuBuffers 함수 구현
bool allocateCpuGpuBuffers(nvinfer1::ICudaEngine* engine, std::vector<void*>& cpuInputBuffers, std::vector<void*>& gpuInputBuffers, std::vector<void*>& cpuOutputBuffers, std::vector<void*>& gpuOutputBuffers) {
    assert(engine != nullptr && "TensorRT engine should not be null.");
    
    size_t returnSize;
    // 입력 및 출력 메모리 할당 및 엔진에 바인딩
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        const auto& binding = engine->getBindingDimensions(i);
        nvinfer1::DataType type = engine->getBindingDataType(i);
        size_t elementSize = getDataTypeSize(type);
        size_t size = volume(binding) * elementSize;
        
        // CPU 메모리 할당
        void* cpuBuffer = static_cast<float*>(malloc(size));
        // GPU 메모리 할당
        void* gpuBuffer;
        cudaMalloc(&gpuBuffer, size);
        
        if (engine->bindingIsInput(i))
        {
            cpuInputBuffers.emplace_back(cpuBuffer);
            gpuInputBuffers.emplace_back(gpuBuffer);
        }
        else
        {
            cpuOutputBuffers.emplace_back(cpuBuffer);
            gpuOutputBuffers.emplace_back(gpuBuffer);
        }
    }
    return true;
}

// allocateCpuGpuBuffers 함수 구현
void** allocateBuffer(nvinfer1::ICudaEngine* engine, bool is_input, bool is_gpu) {
    assert(engine != nullptr && "TensorRT engine should not be null.");
    std::vector<void*> buffers;
    
    // 입력 및 출력 메모리 할당 및 엔진에 바인딩
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        const auto& binding = engine->getBindingDimensions(i);
        nvinfer1::DataType type = engine->getBindingDataType(i);
        size_t elementSize = getDataTypeSize(type);
        size_t size = volume(binding) * elementSize;
        
        if (!(engine->bindingIsInput(i) ^ is_input))
        {   
            if (is_gpu)
            {
                void* gpuBuffer;
                cudaMalloc(&gpuBuffer, size);
                buffers.emplace_back(gpuBuffer);
            }
            else
            {
                void* cpuBuffer = static_cast<float*>(malloc(size));
                buffers.emplace_back(cpuBuffer);
            }
        }
        else continue;
    }
    
    return buffers.data();
}

