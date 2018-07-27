**添加系统变量：`C:\OpenCV\OpenCV342\bin`**

## 以下是新建立一个控制台程序需要添加的内容：

属性->VC++目录->库目录：

```
C:\OpenCV\OpenCV342\lib
```

属性->VC++目录->包含目录：

```
C:\OpenCV\OpenCV342\include
```

### debug版本依赖项：

属性->链接器->输入->附加依赖项：
```
opencv_calib3d342d.lib;opencv_core342d.lib;opencv_dnn342d.lib;opencv_features2d342d.lib;opencv_flann342d.lib;opencv_highgui342d.lib;opencv_imgcodecs342d.lib;opencv_imgproc342d.lib;opencv_ml342d.lib;opencv_objdetect342d.lib;opencv_photo342d.lib;opencv_shape342d.lib;opencv_stitching342d.lib;opencv_superres342d.lib;opencv_ts342d.lib;opencv_video342d.lib;opencv_videoio342d.lib;opencv_videostab342d.lib;
```

## release版本依赖项：

属性->链接器->输入->附加依赖项：
```
opencv_calib3d342.lib;opencv_core342.lib;opencv_dnn342.lib;opencv_features2d342.lib;opencv_flann342.lib;opencv_highgui342.lib;opencv_imgcodecs342.lib;opencv_imgproc342.lib;opencv_ml342.lib;opencv_objdetect342.lib;opencv_photo342.lib;opencv_shape342.lib;opencv_stitching342.lib;opencv_superres342.lib;opencv_ts342.lib;opencv_video342.lib;opencv_videoio342.lib;opencv_videostab342.lib;      
``` 
