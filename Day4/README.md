# Day 4

第四天的作业和第三天类似，只不过是从识别数字变成识别人脸，而且模型什么的要自己训练，算是一个总结吧，差不多把暑期学校期间学到的东西都用上了。

训练模型这块因为图片太多了，都丢去 Github 不太好，所以我只给出我的目录结构: tree.txt ，反正图片文件大家都有 : ) 在 BuildLMDB/ 中的一些文件为了节省空间所以删掉了，但可以自行生成（按顺序）：

```
1. python gen_filename.py   -> training.txt + testing.txt
2. pack.bat                 -> training_mdb/ + testing_mdb/
3. train_model.bat          -> trained/ 
```

有一个 add_noise.py 是用来给图片加椒盐噪声和高斯噪声的，不过我加了之后再训练模型，准确率只是从 0.88 提升到 0.89 而已，提升不是很大

解释一下为什么 recognize_face 函数会出现下面这两行代码，因为在训练的时候 caffe 读入的是**三通道的灰度图像**，这就是为什么 cv::dnn::ConvolutionLayerImpl::getMemoryShapes 会出现断言错误。之前使用手写训练集的时候，我们都是转成了单通道，这样没问题，因为手写训练集就是在单通道的情况下训练的（所以在人脸识别代码出现断言错误时用手写数据集能跑 = = ）。而现在训练出来的模型是三通道的，这样就需要用第一行代码将彩色图像转换成灰度图，然后再用第二行代码将灰度图再转成三通道，这样模型就可以工作了。

```
cvtColor(img, img, COLOR_BGR2GRAY);
cvtColor(img, img, COLOR_GRAY2BGR);
```

PS: 这个作业差不多是我在这期间写得最好的一个作业吧，我指的是代码的结构这些，该放到 header 文件的都放过去了

PPS: 看到这里你大概是已经看完前三天的了吧。记住！看完这个仓库就好了，不要去翻我其他仓库！我的 Github 就像个垃圾堆.... qaq
