#include "stdafx.h"
#include "SummerSchool.h"

void SummerSchool::image_read(string filename)
{
  cout << "函数功能：读入并显示一张图像" << endl;

	const char* imagename = filename.c_str();
	// 从文件中读入图像
  // 灰度: CV_LOAD_IMAGE_GRAYSCALE or 彩色: CV_LOAD_IMAGE_COLOR
	image = imread(imagename, CV_LOAD_IMAGE_COLOR);
	//如果读入图像失败
	if (image.empty())
	{
    cerr << "Can not load image: " << imagename << endl;
		exit(1);
	}
	// 显示图像
	namedWindow("image", 1);
	imshow("image", image);
	// 此函数等待按键，按键盘任意键就返回
	waitKey(0);
}

void SummerSchool::image_reverse()
{
  Mat reversed_image;
  reversed_image.create(image.rows, image.cols, image.type());

  for (int i = 0; i < image.rows; ++i)
  {
    uchar* image_ptr = image.ptr<uchar>(i);
    uchar* reversed_image_ptr = reversed_image.ptr<uchar>(i);
    for (int j = 0; j < image.cols; ++j)
    {
      if(image.channels() == 1)
      {
        reversed_image_ptr[j] = image_ptr[image.cols - j];
      }
      else
      {
        uchar* image_ptr_j = image_ptr + image.elemSize() * j;
        uchar* reversed_image_ptr_j = reversed_image_ptr + image.elemSize() * (image.cols - j);
        reversed_image_ptr_j[0] = image_ptr_j[0]; // B
        reversed_image_ptr_j[1] = image_ptr_j[1]; // G
        reversed_image_ptr_j[2] = image_ptr_j[2]; // R
      }
    }
  }
  imshow("image_reverse", reversed_image);
  waitKey(0);
}

void SummerSchool::image_sobel()
{
  Mat gray;
  if(image.channels() == 1)
    gray = image.clone();
  else
    cvtColor(image, gray, CV_BGR2GRAY);

  /*
  Mat kernel(3, 3, CV_32F, Scalar(0));
  kernel.at<float>(0, 0) = -1.0;
  kernel.at<float>(0, 2) = 1.0;
  kernel.at<float>(1, 0) = -2.0;
  kernel.at<float>(1, 2) = 2.0;
  kernel.at<float>(2, 0) = -1.0;
  kernel.at<float>(2, 2) = 1.0;
  Mat result;
  filter2D(gray, result, CV_8U, kernel, Point(-1, -1));
  //*/

  //* OR
  Mat grad_x, grad_y, abs_grad_x, abs_grad_y, result;
  Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
  Sobel(gray, grad_y, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, result);
  //*/

  imshow("image_sobel", result);
  waitKey(0);
}

void SummerSchool::video_read(string filename)
{
	VideoCapture capture;
	capture.open(filename);

	double rate = capture.get(CV_CAP_PROP_FPS);//获取视频文件的帧率
	int delay = cvRound(1000.000 / rate);

  // prepare face detector
	CascadeClassifier cascade_face_detector;
	string face_cascade_name = "res/haarcascade_frontalface_alt.xml";
	if (!cascade_face_detector.load(face_cascade_name))
	{
		fprintf(stderr, "No classifier!");
		exit(1);
	}
  vector<Rect> FaceVector;
  float searchScaleFator = 1.1f;
  int minNeighbors = 4;
  // int flags = CASCADE_SCALE_IMAGE; // detect multi face
  int flags = CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH; // detect one face
  Size minFeatureSize(10, 10);

	if (!capture.isOpened())//判断是否打开视频文件
	{
		printf("离开\n");
		exit(1);
	}
	else
	{
		Mat frame;
		capture >> frame; //读出每一帧的图像
		while (!frame.empty())
		{
			imshow("处理前视频", frame);
			
      /* face detector start */
      cascade_face_detector.detectMultiScale(frame, FaceVector, searchScaleFator, minNeighbors, flags, minFeatureSize);
      Mat result = frame.clone();
      for (size_t i = 0; i < FaceVector.size(); ++i)
      {
        Rect rect = FaceVector[i];
        rectangle(result, rect, Scalar(0, 255, 0), 2);
      }
      /* face detector end */

			imshow("处理后视频", result);
			waitKey(delay);
			capture >> frame; //读出每一帧的图像
		}
	}
}

void SummerSchool::video_read()
{
	VideoCapture capture(0);

	if (!capture.isOpened()) //检查摄像头是否成功打开
	{
		printf("视频没有读取成功\n");
		exit(0);
	}

	//这里必须两个一起设置才能生效，宽和高
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	//capture.set(CV_CAP_PROP_SHARPNESS, 100);

	Mat frame; //当前视频帧
	int i = 0;
	while (capture.read(frame))
	{
		imshow("video", frame);
		waitKey(30);
		if (++i > 100)
			break;
	}
}

void SummerSchool::detect_face()
{
  // prepare face detector
  CascadeClassifier cascade_face_detector;
  string face_cascade_name = "res/haarcascade_frontalface_alt.xml";
  if (!cascade_face_detector.load(face_cascade_name))
  {
    fprintf(stderr, "No classifier!");
    exit(1);
  }

  vector<Rect> FaceVector;
  float searchScaleFator = 1.1f;
  int minNeighbors = 4;
  // int flags = CASCADE_SCALE_IMAGE; // detect multi face
  int flags = CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH; // detect one face
  Size minFeatureSize(10, 10);
  cascade_face_detector.detectMultiScale(image, FaceVector, searchScaleFator, minNeighbors, flags, minFeatureSize);

  Mat result = image.clone();
  for (size_t i = 0; i < FaceVector.size(); ++i)
  {
    Rect rect = FaceVector[i];
    rectangle(result, rect, Scalar(0, 255, 0), 2);
  }

  imshow("result", result);
  waitKey(0);
}
