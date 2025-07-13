### cpp_rgb2rgb_3dnr + multi_thread + concat_inside_model

- build 폴더가 없다면, 만들고 빌드 및 실행합니다.
```
mkdir build && cd build
cmake ../
make
ENERZAiDemo_multi_thread
```

- build 를 이미 끝냈다면, 아래와 같이 할 수도 있습니다.
```
# build 폴더 밖으로 나옴
cd ..
cmake --build build --target all
build/ENERZAiDemo_multi_thread
```
