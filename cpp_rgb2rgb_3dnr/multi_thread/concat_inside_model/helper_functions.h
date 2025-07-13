#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <NvInfer.h>
#include <NvUffParser.h>
#include <NvUtils.h>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <numeric>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <boost/python.hpp>
#include <ctime>
#include <future>
#include <shared_mutex>

namespace bp = boost::python;

struct Config {
    // camera
    int input_width_ = 1920;
    int input_height_ = 1088;
    int framerate_ = 30;
    int flip_method_ = 0;    
    
    // tensor
    int batch_size_ = 1;
    int channels_ = 3;
    int frame_cnt_ = 7;
    
    // engine
    // concat in model
    std::string engine_path_ = "concat+post+fp32_onnx-fp32_trt-fp16.engine";
    // concat before model 
    // std::string engine_path_ = "postprocessing_onnx-fp32_trt-fp16.engine";
    bool synchronize_stream_ = true;
    
    // loop
    int total_frames_ = 100;
    bool is_multi_thread_ = true;
    bool is_queue_ = true;
    int queue_max_length_ = 1;
    std::string window_name_ = "ENERZAi Demo";
};

// Thread-safe queue 
template <typename T> 
class TSQueue { 
private: 
    // Underlying queue 
    std::queue<T> m_queue; 
  
    // mutex for thread synchronization 
    std::mutex m_mutex; 
  
    // Condition variable for signaling 
    std::condition_variable m_cond1;
    std::condition_variable m_cond2;
    
    // queue max length
    int max_length = 1;
  
public: 
    void setLength(int L)
    {
        max_length = L;
    }

    // Pushes an element to the queue 
    void push(T item) 
    { 
        // Acquire lock 
        std::unique_lock<std::mutex> lock(m_mutex); 
        
        // if full
        // pop the oldest one
        if (m_queue.size() == max_length) m_queue.pop();
        // m_cond2.wait(lock, [this]() { return m_queue.size() != max_length; });
        
        // Add item 
        m_queue.push(item); 
  
        // Notify one thread that 
        // is waiting 
        m_cond1.notify_one(); 
        
        
    } 
  
    // Pops an element off the queue 
    T pop() 
    {
        // acquire lock 
        std::unique_lock<std::mutex> lock(m_mutex); 
  
        // wait until queue is not empty 
        m_cond1.wait(lock, 
                    [this]() { return !m_queue.empty(); }); 
  
        // retrieve item 
        T item = m_queue.front(); 
        m_queue.pop(); 
  	
  	    m_cond2.notify_one();
        // return item 
        return item; 
    } 
}; 

class Logger : public nvinfer1::ILogger {
public:
    // noexcept 예외 명세자를 추가
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // 로깅 수준에 따라 메시지 출력
        if (severity == nvinfer1::ILogger::Severity::kERROR) {
            std::cerr << "Error: " << msg << std::endl;
        } else if (severity == nvinfer1::ILogger::Severity::kWARNING) {
            std::cerr << "Warning: " << msg << std::endl;
        } else {
            std::cout << msg << std::endl;
        }
    }
};

// 헬퍼 함수: 엔진 로드
nvinfer1::ICudaEngine* loadEngine(const std::string& enginePath, nvinfer1::ILogger* logger);

// 헬퍼 함수: gstreamer pipeline text
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method);

// 헬퍼 함수: Dims 정보를 문자열로 변환
std::string dimsToString(const nvinfer1::Dims& dims);

// 헬퍼 함수: Dims 정보를 이용하여 Tensor 크기 계산
size_t volume(const nvinfer1::Dims& dims);

// 헬퍼 함수: 텐서 정보 출력
void printTensorInfo(nvinfer1::ICudaEngine* engine);

// 헬퍼 함수: 열거형을 기반으로 텐서의 데이터 타입 크기를 계산하는 사용자 정의 함수
size_t getDataTypeSize(nvinfer1::DataType type);

// 헬퍼 함수: 버퍼 할당 V2
bool allocateCpuGpuBuffers(nvinfer1::ICudaEngine* engine, std::vector<void*>& cpuInputBuffers, std::vector<void*>& gpuInputBuffers, std::vector<void*>& cpuOutputBuffers, std::vector<void*>& gpuOutputBuffers);

void** allocateBuffer(nvinfer1::ICudaEngine* engine, bool is_input, bool is_gpu);

#endif  // HELPER_FUNCTIONS_H
