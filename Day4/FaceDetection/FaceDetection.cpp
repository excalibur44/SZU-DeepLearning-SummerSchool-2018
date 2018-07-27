// FaceDetection.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "FaceDetection.h"

int main()
{
  initialize();

  /* ==== 读取图片 ==== */
  Mat img = imread(imageFile);
  if (img.empty())
  {
    cerr << "Can't read image from the file: " << imageFile << std::endl;
    exit(1);
  }

  /* ==== 检测人脸 ==== */
  Rect LocalArea = Rect(0, 0, img.cols, img.rows);
  vector<Rect> FaceVector;
  vector<Rect> PreProcessFaceVector;
  detect_face(img, LocalArea, FaceVector, PreProcessFaceVector);

  /* ==== 识别人脸 ==== */
  for (size_t i = 0; i < FaceVector.size(); i++)
  {
    int classId;
    double classProb;
    Mat faceROI = img(FaceVector[i]);
    recognize_face(faceROI, &classId, &classProb);
    rectangle(img, FaceVector[i], Scalar(0, 0, 255), 2);
    int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_scale = 1;
    int thickness = 2;
    int baseline;
    Size text_size = getTextSize(classNames.at(classId), font_face, font_scale, thickness, &baseline);
    Point org(FaceVector[i].x, FaceVector[i].y - text_size.height / 2);
    putText(img, classNames.at(classId), org, font_face, font_scale, Scalar(0, 0, 255), thickness);

    /* ==== 打印结果 ==== */
    cout << "Best class : #" << classId << endl
      << "Class name : " << classNames.at(classId) << endl
      << "Probability: " << classProb * 100 << "%" << endl;
  }

  /* ==== 显示图像 ==== */
  namedWindow("image", 1);
  imshow("image", img);
  waitKey(0);

  return 0;
}

void initialize()
{
  /* ==== 读取分类器 ==== */
  if (!cascade_face_detector.load(cascade_path))
  {
    cerr << "No classifier!" << endl;
    exit(1);
  }

  /* ==== 读取模型参数和模型结构文件 ==== */
  net = dnn::readNetFromCaffe(modelTxt, modelBin); //合成网络
  if (net.empty())
  {
    cerr << "Can't load network!" << endl;
    exit(1);
  }

  /* ==== 从标签文件读取分类 空格为标志 ==== */
  ifstream fp(labels_filename);
  if (!fp.is_open())
  {
    cerr << "File with classes labels not found: " << labels_filename << endl;
    exit(-1);
  }
  string name;
  while (!fp.eof())
  {
    getline(fp, name);
    if (name.length())
      classNames.push_back(name.substr(name.find(' ') + 1));
  }
  fp.close();
}

void detect_face(Mat image, Rect LocalArea, vector<Rect> &FaceVector, vector<Rect> &PreProcessFaceVector)
{
  Mat ImgROI = image(LocalArea); // 直接把感兴趣区域拷贝出来

  Size minFeatureSize(10, 10);
  float searchScaleFactor = 1.1f;
  int minNeighbors = 4;
  int flags = CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH;
  // CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH 只检测脸最大的人 
  // or CASCADE_SCALE_IMAGE; // 检测多个人
  cascade_face_detector.detectMultiScale(ImgROI, FaceVector, searchScaleFactor, minNeighbors, flags, minFeatureSize);

  printf("人脸个数: %d\n", FaceVector.size());
  if (FaceVector.size()>0)
  {
    Rect *object_rect = new Rect[FaceVector.size()];
    //局部改为全局的坐标
    for (size_t i = 0; i<FaceVector.size(); i++)
    {
      FaceVector[i].x = FaceVector[i].x + LocalArea.x;
      FaceVector[i].y = FaceVector[i].y + LocalArea.y;

      object_rect[i] = FaceVector[i];
    }

    //对检测到的人脸按照面积大小进行排序
    //冒泡排序，把最小的放最后面
    for (size_t i = 1; i <= FaceVector.size(); i++)//最多运行FaceVector.size()-1次
    {
      for (size_t j = 1; j <= FaceVector.size() - i; j++)
      {
        if (object_rect[j - 1].height * object_rect[j - 1].width < object_rect[j].height * object_rect[j].width)
        {
          Rect tmp;
          tmp = object_rect[j - 1];
          object_rect[j - 1] = object_rect[j];
          object_rect[j] = tmp;
        }
      }
    }
    
    //FaceVector和object_rect同样数量矩形框
    //最多取前10个存入PreProcessFaceVector
    int FaceNum = 0;
    for (size_t i = 0; i < FaceVector.size(); ++i, ++FaceNum)
    {
      PreProcessFaceVector.push_back(object_rect[i]);
    }
    //printf("人脸个数%d\n", PreProcessFaceVector.size());

    if (object_rect != NULL)
    {
      delete[] object_rect;
      object_rect = NULL;
    }
  }
}

void recognize_face(Mat src_img, int *classId, double *classProb)
{
  Mat img = src_img.clone();
  resize(img, img, Size(28, 28), 0, 0, INTER_LINEAR);
  cvtColor(img, img, COLOR_BGR2GRAY);
  cvtColor(img, img, COLOR_GRAY2BGR);

  // 构造 blob ，为传入网络做准备，图片不能直接进入网络
  // 必须要归一化 : 1.0 / 256
  Mat inputBlob = blobFromImage(img, 1.0 / 256, Size(28, 28));

  Mat prob;
  TickMeter t;
  for (int i = 0; i < 1; i++)
  {
    CV_TRACE_REGION("forward");
    //将构建的blob传入网络data层
    net.setInput(inputBlob, "data");
    //计时
    t.start();
    //前向预测
    prob = net.forward("prob");
    //停止计时
    t.stop();
  }
  //打印出花费时间
  //cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << endl;

  //找出最高的概率ID存储在classId，对应的标签在classProb中
  //getMaxClass(prob, classId, classProb);
  Mat probMat = prob.reshape(1, 1);
  Point classNumber;
  minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
  *classId = classNumber.x;
}
