### python_raw2raw_2dnr + single_thread + isp_inside_model

- torch 모델의 onnx 파일이 없다면, 모델을 변경하고 checkpoint 를 onnx 파일로 변환합니다.
```
gedit model.py
python torch2onnx.py
```

- torch 모델의 tensorrt engine 파일이 없다면, 빌드합니다. 
```
./onnx2trt.sh
```

- 실행합니다.
```
python pycuda_raw2raw.py
```
