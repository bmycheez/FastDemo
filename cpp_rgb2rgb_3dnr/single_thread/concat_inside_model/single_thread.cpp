#include "helper_functions.h"
#include "cuda_functions.h"

#define pass (void)0

#pragma region TensorRT
Logger gLogger;
Config config;
cudaStream_t stream;
nvinfer1::ILogger& logger = gLogger;
nvinfer1::ILogger* trtLogger = &logger;
nvinfer1::ICudaEngine* engine = loadEngine(config.engine_path_, trtLogger);
nvinfer1::IExecutionContext* context = engine->createExecutionContext();
#pragma endregion

#pragma region gstreamer
std::string pipeline = gstreamer_pipeline(
    config.input_width_,
    config.input_height_,
    config.input_width_,
    config.input_height_,
    config.framerate_,
    config.flip_method_);
cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
#pragma endregion

#pragma region multithread
std::atomic<int> i(config.total_frames_);
#pragma endregion

#pragma region latency
std::vector<float> runtime1;
std::vector<float> runtime2;
std::vector<float> runtime3;
std::vector<float> runtime4;
std::vector<float> runtime5;
std::vector<float> runtime6;
std::vector<float> runtime7;
std::vector<float> runtime8;
std::chrono::system_clock::time_point startTime5;

auto startTime1 = std::chrono::system_clock::now();
auto myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
auto endTime1 = std::chrono::system_clock::now();
auto latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
#pragma endregion

size_t singleInputSize = config.batch_size_ * config.channels_ * config.input_height_ * config.input_width_ * sizeof(float);
size_t singleOutputSize = singleInputSize;
std::vector<void*> cpuInputBuffers;
std::vector<void*> gpuInputBuffers;
std::vector<void*> cpuOutputBuffers;
std::vector<void*> gpuOutputBuffers;
bool buffer_ready = allocateCpuGpuBuffers(engine, cpuInputBuffers, gpuInputBuffers, cpuOutputBuffers, gpuOutputBuffers);

void Pipeline()
{
    cv::Mat hwcFrame;
    int copied = 0;
    cv::Mat display(config.input_height_, config.input_width_, CV_32FC3);
    std::vector<float> result(config.input_height_ * config.input_width_ * config.channels_);
    
    while(i > 0)
    {
        startTime1 = std::chrono::system_clock::now();
        myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th Read Start!: " << -myTime1 << std::endl;
    	#pragma region Read
    	if (!cap.read(hwcFrame)) {
		    std::cout << "Capture read error!" << std::endl;
		    break;
	    }
	    #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th Read End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime1.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
        myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th HWC > CHW Start!: " << -myTime1 << std::endl;
    	#pragma region HWC > CHW
	    int size[] = {config.channels_, hwcFrame.rows, hwcFrame.cols};
	    cv::Mat bchwFrame(config.channels_, size, CV_8U);
	    std::vector<cv::Mat> planes = {
	        cv::Mat(hwcFrame.rows, hwcFrame.cols, CV_8U, bchwFrame.ptr(0)),
	        cv::Mat(hwcFrame.rows, hwcFrame.cols, CV_8U, bchwFrame.ptr(1)),
	        cv::Mat(hwcFrame.rows, hwcFrame.cols, CV_8U, bchwFrame.ptr(2))
	    };
	    cv::split(hwcFrame, planes);
	    #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th HWC > CHW End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime2.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
        myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th change dtype Start!: " << -myTime1 << std::endl;
	    #pragma region change dtype
	    bchwFrame.convertTo(bchwFrame, CV_32F);
	    #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th change dtype End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime3.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
        myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th H2D copy Start!: " << -myTime1 << std::endl;
	    #pragma region H2D copy
	    if (copied < config.frame_cnt_)
	    {
	        cudaMemcpyAsync(gpuInputBuffers[copied], bchwFrame.data, singleInputSize, cudaMemcpyHostToDevice, stream);
	        copied += 1;
	        i--;
	        continue;
	    }
	    else
	    {
	        gpuInputBuffers.insert(gpuInputBuffers.end(), gpuInputBuffers[0]);
	        gpuInputBuffers.erase(gpuInputBuffers.begin());
	        cudaMemcpyAsync(gpuInputBuffers.back(), bchwFrame.data, singleInputSize, cudaMemcpyHostToDevice, stream);
	    }
	    void* bindings[] = {gpuInputBuffers[0], gpuInputBuffers[1], gpuInputBuffers[2], gpuInputBuffers[3], gpuInputBuffers[4], gpuInputBuffers[5], gpuInputBuffers[6], gpuOutputBuffers[0]};
	    #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th H2D copy End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime4.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th Model Start!: " << -myTime1 << std::endl;
	    #pragma region Model
        context->enqueueV2(bindings, stream, nullptr);
        if (config.synchronize_stream_) cudaStreamSynchronize(stream);
	    #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th Model End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime5.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th D2H copy Start!: " << -myTime1 << std::endl;
	    #pragma region D2H copy
        cudaMemcpyAsync(result.data(), gpuOutputBuffers[0], singleOutputSize, cudaMemcpyDeviceToHost, stream);
        #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th D2H copy End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime6.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th CHW > HWC Start!: " << -myTime1 << std::endl;
	    #pragma region CHW > HWC
        cv::Mat B(config.input_height_, config.input_width_, CV_32FC1, result.data());
        cv::Mat G(config.input_height_, config.input_width_, CV_32FC1, result.data() + config.input_height_ * config.input_width_);
        cv::Mat R(config.input_height_, config.input_width_, CV_32FC1, result.data() + 2 * config.input_height_ * config.input_width_);
        std::vector<cv::Mat> channels{B, G, R};
        cv::merge(channels, display);
        #pragma endregion
        endTime1 = std::chrono::system_clock::now();
        myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th CHW > HWC End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime7.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th Display Start!: " << -myTime1 << std::endl;
	    #pragma region Display
        cv::imshow(config.window_name_, display);
        int keycode = cv::waitKey(1) & 0xff; 
        if (keycode == 27) break;
        i--;
        #pragma endregion
        endTime1 = std::chrono::system_clock::now();
        myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th Display End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime8.emplace_back(latency1.count());
    }
    return;
}

int main(int argc, char* argv[]) 
{   
    #pragma region pre-setting
    cv::namedWindow(config.window_name_, cv::WINDOW_AUTOSIZE);
    if(!cap.isOpened()) 
    {
        std::cout << "Failed to open camera." << std::endl;
        return -1;
    }
    cudaStreamCreate(&stream);
    #pragma endregion
    
    #pragma region main
    startTime5 = std::chrono::system_clock::now();
    Pipeline();
    auto endTime5 = std::chrono::system_clock::now();
    #pragma endregion
    
    #pragma region post-setting
    cap.release();
    cv::destroyAllWindows();
    context->destroy();
    engine->destroy();
    #pragma endregion
    
    #pragma region latency
    float mean1 = std::accumulate(runtime1.begin(), runtime1.end(), 0.0) / runtime1.size();
    float mean2 = std::accumulate(runtime2.begin(), runtime2.end(), 0.0) / runtime2.size();
    float mean3 = std::accumulate(runtime3.begin(), runtime3.end(), 0.0) / runtime3.size();
    float mean4 = std::accumulate(runtime4.begin(), runtime4.end(), 0.0) / runtime4.size();
    float mean5 = std::accumulate(runtime5.begin(), runtime5.end(), 0.0) / runtime5.size();
    float mean6 = std::accumulate(runtime6.begin(), runtime6.end(), 0.0) / runtime6.size();
    float mean7 = std::accumulate(runtime7.begin(), runtime7.end(), 0.0) / runtime7.size();
    float mean8 = std::accumulate(runtime8.begin(), runtime8.end(), 0.0) / runtime8.size();
    auto latency5 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime5 - startTime5);
    float meanT = float(latency5.count()) / (config.total_frames_ - i);
    
    std::cout << config.total_frames_ - i << " frames average: " << std::endl;
    std::cout << "Read: " << mean1 << " (ms) / " << 1000. / mean1 << " (fps)" << std::endl;
    std::cout << "HWC > CHW: " << mean2 << " (ms) / " << 1000. / mean2 << " (fps)" << std::endl;
    std::cout << "change dtype: " << mean3 << " (ms) / " << 1000. / mean3 << " (fps)" << std::endl;
    std::cout << "H2D copy: " << mean4 << " (ms) / " << 1000. / mean4 << " (fps)" << std::endl;
    std::cout << "Model: " << mean5 << " (ms) / " << 1000. / mean5 << " (fps)" << std::endl;
    std::cout << "D2H copy: " << mean6 << " (ms) / " << 1000. / mean6 << " (fps)" << std::endl;
    std::cout << "CHW > HWC: " << mean7 << " (ms) / " << 1000. / mean7 << " (fps)" << std::endl;
    std::cout << "Display: " << mean8 << " (ms) / " << 1000. / mean8 << " (fps)" << std::endl;
    std::cout << "Total: " << meanT << " (ms) / " << 1000. / meanT << " (fps)" << std::endl;
    #pragma endregion
    return 0;
}

