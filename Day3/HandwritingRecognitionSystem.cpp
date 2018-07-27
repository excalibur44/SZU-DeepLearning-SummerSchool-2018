// HandwritingRecognitionSystem.cpp : �������̨Ӧ�ó������ڵ㡣
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
int Control = 0;//�������Ŀ���д�Ͳ�������

int main()
{
  CV_TRACE_FUNCTION();

  /* ==== ��ʼ�����硢ͼƬ������ ==== */
  Net net;
  Mat img;
  vector<String> classNames;
  initialize(&net, &img, &classNames);

  int key = -1;
  //ֻ��image_show�������ƺͲ���������do��if������ʹ����ͼ���Ͽ���ʹ�ü���������Ϊ
  do
  {
    //����������ǣ�ʲô״̬������ʲô���ڽ�������
    imshow("Board", DispImg);

    key = waitKey(0);
    if (key == '1')
    {
      //�ָ� & ʶ��
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
          /* ==== ���� 1 ���ݵ㼯��������ͼ ==== */
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

          /* ==== ���� 2 �����ֿٳ����ŵ�һ�Ŵ�ͼ������ ==== */
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

          /* ==== ����Ԥ�� ==== */
          int classId;
          double classProb;
          recognizeNumber(net, tmpBoard, &classId, &classProb);

          /* ==== ��ӡ��� ==== */
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
      //���̿���д�͹ر�д��״̬
      printf("w\n");
    }
    else if (key == 'q')
    {
      //����2���뿪
      printf("leave\n");
      break;
    }
  } while (1);

  return 0;
}

void initialize(Net *net, Mat* img, vector<String>* classNames)
{
  /* ==== ��ȡģ�Ͳ�����ģ�ͽṹ�ļ� ==== */
  *net = dnn::readNetFromCaffe(modelTxt, modelBin); //�ϳ�����
  if (net->empty())
  {
    cerr << "Can't load network by using the following files: " << std::endl;
    exit(1);
  }
  //cout << "net read successfully" << endl;

  /* ==== ��ȡͼƬ ==== */
  *img = imread(imageFile, 0); //�������Ҫ������Ҫ��0��������3ͨ��������󣡣���
  if (img->empty())
  {
    cerr << "Can't read image from the file: " << imageFile << std::endl;
    exit(1);
  }
  //cout << "image read sucessfully" << endl;

  /* ==== �ӱ�ǩ�ļ���ȡ���� �ո�Ϊ��־ ==== */
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

  /* ==== ����һ����д�� ==== */
  MyBoard.create(height, width, CV_8U);
  DispImg.create(height, width, CV_8U);
  MyBoard.setTo(0);
  MyBoard.copyTo(DispImg); //��һ�θ�ֵ��DispImg

  namedWindow("Board", 1);
  setMouseCallback("Board", mouse_call_func);
}

void recognizeNumber(Net net, Mat img, int *classId, double *classProb)
{
  Mat img_new;
  img_new.create(img.rows, img.cols, img.type());
  //��ͼ����ж�ֵ������
  threshold(img, img_new, 100, 255, THRESH_BINARY); //img.copyTo(img_new);

  //����blob��Ϊ����������׼����ͼƬ����ֱ�ӽ�������
  /*	Mat inputBlob = blobFromImage(img, 1, Size(224, 224), Scalar(104, 117, 123)); */
  Mat inputBlob = blobFromImage(img_new, 1.0 / 255, Size(28, 28));// ����Ҫ��һ�� : 1.0 / 255

  Mat prob;
  TickMeter t;
  for (int i = 0; i < 1; i++)   //for (int i = 0; i < 10; i++)   
  {
    CV_TRACE_REGION("forward");
    //��������blob��������data��
    net.setInput(inputBlob, "data");
    //��ʱ
    t.start();
    //ǰ��Ԥ��
    prob = net.forward("prob");
    //ֹͣ��ʱ
    t.stop();
  }
  //��ӡ������ʱ��
  //cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << endl;

  //�ҳ���ߵĸ���ID�洢��classId����Ӧ�ı�ǩ��classProb��
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
  case CV_EVENT_LBUTTONDOWN://����һ����
    CurrentPosition = Point(x, y);

    printf("��ǰ�����꣺(%d,%d)\n", CurrentPosition.x, CurrentPosition.y);

    if (Control == 0)//��ԭ�����ڿ�״̬���ͱ�Ϊд״̬
    {
      Control = 1;
      printf("����������д��\n");
    }
    break;
  case CV_EVENT_LBUTTONUP:
    printf("�ͷ����\n");
    if (Control == 1)//��ԭ������д��״̬���ͱ�ֹͣд��
    {
      Control = 0;
      printf("������ֹͣд��\n");
      //������������ 
    }
    break;
  case CV_EVENT_RBUTTONDOWN://�������ܣ��������һ������
    printf("�ͷ��Ҽ�\n");
    if (Control == 1 || Control == 0)//��ԭ������д��״̬���ͱ�ֹͣд��
    {
      Control = 2;
      printf("�Ҽ�ֹͣ�����д�֣����������������һ��\n");
    }
    break;
  }

  if (Control == 1)//��ʼд��
  {
    printf("д��ing, (%d,%d)\n", CurrentPosition.x, CurrentPosition.y);

    //����һ�µ�ǰλ�õ�ֵ
    PreviousPosition = CurrentPosition;
    CurrentPosition = Point(x, y);
    if (0 < CurrentPosition.x - 2
      && CurrentPosition.x + 2 < width
      && 0 < CurrentPosition.y - 2
      && CurrentPosition.y + 2 < height)
    {
      //�ѵ�ǰλ�õĸ���3��������Ϊ��ɫ������ʾ
      DispImg.at<uchar>(y, x) = 255;
      DispImg.at<uchar>(y - 1, x - 1) = 255;
      DispImg.at<uchar>(y - 1, x) = 255;
      DispImg.at<uchar>(y - 1, x + 1) = 255;
      DispImg.at<uchar>(y, x - 1) = 255;
      DispImg.at<uchar>(y, x + 1) = 255;
      DispImg.at<uchar>(y + 1, x - 1) = 255;
      DispImg.at<uchar>(y + 1, x) = 255;
      DispImg.at<uchar>(y + 1, x + 1) = 255;
      //��ǰ���֮ǰ��������
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
    //�Ƴ����һ�Σ������㣻 ��û�У��Ͳ���������Ҳ����
    Control = 0;
    MyBoard.copyTo(DispImg);

    //������������
    printf("�Ƴ����һ�Σ������㣻 ��û�У��Ͳ���������Ҳ����\n");
  }
  imshow("Board", DispImg);
}
