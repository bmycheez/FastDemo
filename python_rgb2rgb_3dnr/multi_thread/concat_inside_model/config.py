# demo config
inp_shape = (1, 21, 1088, 1920)
outp_shape = (1, 3, 1088, 1920)
gst_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1088, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)1920, height=(int)1088, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
frame_cnt = 7
engine_path = 'concat+post+fp32_onnx-fp32_trt-fp16.engine'

# model_config
device_num = 0
queue_size = 1
total_frames = 1200
display_input = False
