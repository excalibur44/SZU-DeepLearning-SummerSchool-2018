// HandwritingRecognitionSystem.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

/* ==== function ==== */
void initialize(Net *net, Mat* img, vector<String>* classNames);
void recognizeNumber(Net net, Mat img, int *classId, double *classProb);
static void  mouse_call_func(int event, int x, int y, int flags, void* param);

/* ==== metadata ==== */
static String modelTxt = "res/deploy.prototxt";
static String modelBin = "res/lenet_iter_10000.caffemodel";
static String imageFile = "res/44.jpg";
static char* labels_filename = "res/labels.txt";
int width = 1200;
int height = 600;
Mat MyBoard;
Mat DispImg;
Point PreviousPosition = Point(0, 0);
Point CurrentPosition;
int Control = 0;//这是鼠标的控制写和擦除变量

int main()
{
  CV_TRACE_FUNCTION();

  /* ==== 初始化网络、图片和类名 ==== */
  Net net;
  Mat img;
  vector<String> classNames;
  initialize(&net, &img, &classNames);

  int key = -1;
  //只对image_show进行限制和操作，引入do和if操作，使得在图像上可以使用键盘输入作为
  do
  {
    //这里做个标记，什么状态，就向什么窗口进行输入
    imshow("Board", DispImg);

    key = waitKey(0);
    if (key == '1')
    {
      //分割 & 识别
      printf("split\n");
      vector<vector<Point>> contours;
      findContours(DispImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

      printf("recognize\n");
      for (size_t i = 0; i < contours.size(); i++)
      {
        vector<Point> points = contours[i];
        Rect r = boundingRect(points);
        if (width * height * 0.0001 < r.width * r.height && r.width * r.height < width * height)
        {
          /* ==== 方法 1 根据点集生成数字图 ==== */
          Mat drawBoard;
          drawBoard.create(r.height + 2, r.width + 2, CV_8U);
          drawBoard.setTo(0);
          for (size_t j = 1; j < points.size(); j++)
          {
            line(drawBoard,
              Point(points[j - 1].x - r.x + 1, points[j - 1].y - r.y + 1),
              Point(points[j].x - r.x + 1, points[j].y - r.y + 1),
              Scalar(255), 6);
          }
          //imshow("drawBoard", drawBoard);
          int max_length = int(((r.width > r.height) ? r.width + 2 : r.height + 2) * 1.40);
          Mat tmpBoard;
          tmpBoard.create(max_length, max_length, CV_8U);
          tmpBoard.setTo(0);
          Mat tmpBoardROI = tmpBoard(Rect((max_length - drawBoard.cols) / 2, (max_length - drawBoard.rows) / 2, drawBoard.cols, drawBoard.rows));
          drawBoard.copyTo(tmpBoardROI);
          resize(tmpBoard, tmpBoard, Size(28, 28), 0, 0, INTER_LINEAR);

          /* ==== 方法 2 将数字抠出来放到一张大图的中央 ==== */
          /*
          Mat DispImgROI = DispImg(r);
          int max_length = int(((r.width > r.height) ? r.width : r.height) * 1.40);
          Mat tmpBoard;
          tmpBoard.create(max_length, max_length, CV_8U);
          tmpBoard.setTo(0);
          Mat tmpBoardROI = tmpBoard(Rect((max_length - r.width) / 2, (max_length - r.height) / 2, r.width, r.height));
          DispImgROI.copyTo(tmpBoardROI);
          resize(tmpBoard, tmpBoard, Size(28, 28), 0, 0, INTER_LINEAR);
          */

          /* ==== 进行预测 ==== */
          int classId;
          double classProb;
          recognizeNumber(net, tmpBoard, &classId, &classProb);

          /* ==== 打印结果 ==== */
          cout << "Best class : #" << classId << endl
            << "Class name : " << classNames.at(classId) << endl
            << "Probability: " << classProb * 100 << "%" << endl;
          imshow("image", tmpBoard);
          waitKey(0);
          destroyWindow("image");
        }
      }
    }
    else if (key == 'w' || key == 'W')
    {
      //键盘开启写和关闭写的状态
      printf("w\n");
    }
    else if (key == 'q')
    {
      //等于2则离开
      printf("leave\n");
      break;
    }
  } while (1);

  return 0;
}

void initialize(Net *net, Mat* img, vector<String>* classNames)
{
  /* ==== 读取模型参数和模型结构文件 ==== */
  *net = dnn::readNetFromCaffe(modelTxt, modelBin); //合成网络
  if (net->empty())
  {
    cerr << "Can't load network by using the following files: " << std::endl;
    exit(1);
  }
  //cout << "net read successfully" << endl;

  /* ==== 读取图片 ==== */
  *img = imread(imageFile, 0); //这个很重要，必须要加0，否则变成3通道，会错误！！！
  if (img->empty())
  {
    cerr << "Can't read image from the file: " << imageFile << std::endl;
    exit(1);
  }
  //cout << "image read sucessfully" << endl;

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
      classNames->push_back(name.substr(name.find(' ') + 1));
  }
  fp.close();

  /* ==== 创建一张手写板 ==== */
  MyBoard.create(height, width, CV_8U);
  DispImg.create(height, width, CV_8U);
  MyBoard.setTo(0);
  MyBoard.copyTo(DispImg); //第一次赋值给DispImg

  namedWindow("Board", 1);
  setMouseCallback("Board", mouse_call_func);
}

void recognizeNumber(Net net, Mat img, int *classId, double *classProb)
{
  Mat img_new;
  img_new.create(img.rows, img.cols, img.type());
  //对图像进行二值化处理
  threshold(img, img_new, 100, 255, THRESH_BINARY); //img.copyTo(img_new);

  //构造blob，为传入网络做准备，图片不能直接进入网络
  /*	Mat inputBlob = blobFromImage(img, 1, Size(224, 224), Scalar(104, 117, 123)); */
  Mat inputBlob = blobFromImage(img_new, 1.0 / 255, Size(28, 28));// 必须要归一化 : 1.0 / 255

  Mat prob;
  TickMeter t;
  for (int i = 0; i < 1; i++)   //for (int i = 0; i < 10; i++)   
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

static void  mouse_call_func(int event, int x, int y, int flags, void* param)
{
  if (DispImg.empty())
    return;
  switch (event)
  {
  case CV_EVENT_LBUTTONDOWN://输入一个点
    CurrentPosition = Point(x, y);

    printf("当前点坐标：(%d,%d)\n", CurrentPosition.x, CurrentPosition.y);

    if (Control == 0)//若原来处于空状态，就变为写状态
    {
      Control = 1;
      printf("鼠标左键开启写字\n");
    }
    break;
  case CV_EVENT_LBUTTONUP:
    printf("释放左键\n");
    if (Control == 1)//若原来处于写字状态，就变停止写字
    {
      Control = 0;
      printf("鼠标左键停止写字\n");
      //最近点清零操作 
    }
    break;
  case CV_EVENT_RBUTTONDOWN://撤销功能，撤销最近一个操作
    printf("释放右键\n");
    if (Control == 1 || Control == 0)//若原来处于写字状态，就变停止写字
    {
      Control = 2;
      printf("右键停止了左键写字，并且启动擦除最近一次\n");
    }
    break;
  }

  if (Control == 1)//开始写字
  {
    printf("写字ing, (%d,%d)\n", CurrentPosition.x, CurrentPosition.y);

    //更新一下当前位置的值
    PreviousPosition = CurrentPosition;
    CurrentPosition = Point(x, y);
    if (0 < CurrentPosition.x - 2
      && CurrentPosition.x + 2 < width
      && 0 < CurrentPosition.y - 2
      && CurrentPosition.y + 2 < height)
    {
      //把当前位置的附近3个像素置为白色，并显示
      DispImg.at<uchar>(y, x) = 255;
      DispImg.at<uchar>(y - 1, x - 1) = 255;
      DispImg.at<uchar>(y - 1, x) = 255;
      DispImg.at<uchar>(y - 1, x + 1) = 255;
      DispImg.at<uchar>(y, x - 1) = 255;
      DispImg.at<uchar>(y, x + 1) = 255;
      DispImg.at<uchar>(y + 1, x - 1) = 255;
      DispImg.at<uchar>(y + 1, x) = 255;
      DispImg.at<uchar>(y + 1, x + 1) = 255;
      //当前点和之前点连成线
      if (0 < PreviousPosition.x - 2
        && PreviousPosition.x + 2 < width
        && 0 < PreviousPosition.y - 2
        && PreviousPosition.y + 2 < height)
      {
        line(DispImg, PreviousPosition, CurrentPosition, Scalar(255), 6);
      }
    }
  }
  else if (Control == 2)
  {
    //移除最近一次，且置零； 若没有，就不操作，但也置零
    Control = 0;
    MyBoard.copyTo(DispImg);

    //最近点清零操作
    printf("移除最近一次，且置零； 若没有，就不操作，但也置零\n");
  }
  imshow("Board", DispImg);
}
