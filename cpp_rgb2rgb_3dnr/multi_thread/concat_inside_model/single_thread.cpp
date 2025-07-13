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
allocateCpuGpuBuffers(engine, config.batch_size_, cpuInputBuffers, gpuInputBuffers, cpuOutputBuffers, gpuOutputBuffers);

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
	    int size[] = {3, hwcFrame.rows, hwcFrame.cols};
	    cv::Mat bchwFrame(3, size, CV_8U);
	    std::vector<cv::Mat> planes = {
	        cv::Mat(hwcFrame.rows, hwcFrame.cols, CV_8U, bchwFrame.ptr(0)),
	        cv::Mat(hwcFrame.rows, hwcFrame.cols, CV_8U, bchwFrame.ptr(1)),
	        cv::Mat(hwcFrame.rows, hwcFrame.cols, CV_8U, bchwFrame.ptr(2))
	    };
	    cv::split(hwcFrame, planes);
	    bchwFrame.convertTo(bchwFrame, CV_32F);
	    #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th Read End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime1.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th H2D Start!: " << -myTime1 << std::endl;
	    #pragma region H2D
	    if (copied < config.frame_cnt_)
	    {
	        cudaMemcpyAsync(gpuInputBuffers[copied], bchwFrame.data, singleInputSize, cudaMemcpyHostToDevice, stream);
	        copied += 1;
	        i--;
	        continue;
	    }
	    else
	        cudaMemcpyAsync(gpuInputBuffers.back(), bchwFrame.data, singleInputSize, cudaMemcpyHostToDevice, stream);
	    #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th H2D End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime2.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th Model Start!: " << -myTime1 << std::endl;
	    #pragma region Model
        void* buffers[] = {gpuInputBuffers[0], gpuInputBuffers[1], gpuInputBuffers[2], gpuInputBuffers[3], gpuInputBuffers[4], gpuInputBuffers[5], gpuInputBuffers[6], gpuOutputBuffers[0]};
        context->enqueueV2(buffers, stream, nullptr);
        if (config.synchronize_stream_) cudaStreamSynchronize(stream);
        gpuInputBuffers.insert(gpuInputBuffers.end(), gpuInputBuffers[0]);
	    gpuInputBuffers.erase(gpuInputBuffers.begin());
	    #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th Model End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime3.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th D2H Start!: " << -myTime1 << std::endl;
	    #pragma region D2H
        cudaMemcpyAsync(result.data(), gpuOutputBuffers[0], singleOutputSize, cudaMemcpyDeviceToHost, stream);
        #pragma endregion
	    endTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th D2H End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime4.emplace_back(latency1.count());
	    
	    startTime1 = std::chrono::system_clock::now();
	    myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - startTime1).count();
        std::cout << config.total_frames_ - i << " th Display Start!: " << -myTime1 << std::endl;
	    #pragma region Display
        cv::Mat B(config.input_height_, config.input_width_, CV_32FC1, result.data());
        cv::Mat G(config.input_height_, config.input_width_, CV_32FC1, result.data() + config.input_height_ * config.input_width_);
        cv::Mat R(config.input_height_, config.input_width_, CV_32FC1, result.data() + 2 * config.input_height_ * config.input_width_);
        std::vector<cv::Mat> channels{B, G, R};
        cv::merge(channels, display);
        cv::imshow(config.window_name_, display);
        int keycode = cv::waitKey(1) & 0xff; 
        if (keycode == 27) break;
        #pragma endregion
        endTime1 = std::chrono::system_clock::now();
        myTime1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(startTime5 - endTime1).count();
        std::cout << config.total_frames_ - i << " th Display End!: " << -myTime1 << std::endl;
	    latency1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	    runtime5.emplace_back(latency1.count());
	    
        i--;
    }
    return;
}

int main(int argc, char* argv[]) 
{   
    #pragma region pre-setting
    startTime5 = std::chrono::system_clock::now();
    cv::namedWindow(config.window_name_, cv::WINDOW_AUTOSIZE);
    if(!cap.isOpened()) 
    {
        std::cout << "Failed to open camera." << std::endl;
        return -1;
    }
    cudaStreamCreate(&stream);
    #pragma endregion
    
    #pragma region main
    Pipeline();
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
    auto endTime5 = std::chrono::system_clock::now();
    auto latency5 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime5 - startTime5);
    float mean6 = float(latency5.count()) / (config.total_frames_ - i);
    
    std::cout << config.total_frames_ - i << " frames average: " << std::endl;
    std::cout << "Read: " << mean1 << " (ms) / " << 1000. / mean1 << " (fps)" << std::endl;
    std::cout << "H2D: " << mean2 << " (ms) / " << 1000. / mean2 << " (fps)" << std::endl;
    std::cout << "Model: " << mean3 << " (ms) / " << 1000. / mean3 << " (fps)" << std::endl;
    std::cout << "D2H: " << mean4 << " (ms) / " << 1000. / mean4 << " (fps)" << std::endl;
    std::cout << "Display: " << mean5 << " (ms) / " << 1000. / mean5 << " (fps)" << std::endl;
    std::cout << "Total: " << mean6 << " (ms) / " << 1000. / mean6 << " (fps)" << std::endl;
    #pragma endregion
    return 0;
}

