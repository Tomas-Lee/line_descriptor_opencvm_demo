# 目标
由于OpenCV3.4.6及以后版本将LSD线特征提取算法从代码中剔除，本库通过对OpenCV3.4.2及OpenCV-Contrib3.4.2的源代码中代码提取，提取出单独的LSD线特征提取算法并封装在cvm命名空间中。

# 使用方式
先进入line_descriptor_opencvm文件夹，然后
```
mkdir build
cd build
cmake ..
make
```
之后会获得相应编译出的LSD算法的头文件和共享库文件
然后对main函数进行cmake编译运行即可。就能得到两张图像的线特征提取，匹配的结果。
