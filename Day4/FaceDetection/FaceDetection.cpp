// FaceDetection.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "FaceDetection.h"

int main()
{
  initialize();

  /* ==== ��ȡͼƬ ==== */
  Mat img = imread(imageFile);
  if (img.empty())
  {
    cerr << "Can't read image from the file: " << imageFile << std::endl;
    exit(1);
  }

  /* ==== ������� ==== */
  Rect LocalArea = Rect(0, 0, img.cols, img.rows);
  vector<Rect> FaceVector;
  vector<Rect> PreProcessFaceVector;
  detect_face(img, LocalArea, FaceVector, PreProcessFaceVector);

  /* ==== ʶ������ ==== */
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

    /* ==== ��ӡ��� ==== */
    cout << "Best class : #" << classId << endl
      << "Class name : " << classNames.at(classId) << endl
      << "Probability: " << classProb * 100 << "%" << endl;
  }

  /* ==== ��ʾͼ�� ==== */
  namedWindow("image", 1);
  imshow("image", img);
  waitKey(0);

  return 0;
}

void initialize()
{
  /* ==== ��ȡ������ ==== */
  if (!cascade_face_detector.load(cascade_path))
  {
    cerr << "No classifier!" << endl;
    exit(1);
  }

  /* ==== ��ȡģ�Ͳ�����ģ�ͽṹ�ļ� ==== */
  net = dnn::readNetFromCaffe(modelTxt, modelBin); //�ϳ�����
  if (net.empty())
  {
    cerr << "Can't load network!" << endl;
    exit(1);
  }

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
      classNames.push_back(name.substr(name.find(' ') + 1));
  }
  fp.close();
}

void detect_face(Mat image, Rect LocalArea, vector<Rect> &FaceVector, vector<Rect> &PreProcessFaceVector)
{
  Mat ImgROI = image(LocalArea); // ֱ�ӰѸ���Ȥ���򿽱�����

  Size minFeatureSize(10, 10);
  float searchScaleFactor = 1.1f;
  int minNeighbors = 4;
  int flags = CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH;
  // CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH ֻ����������� 
  // or CASCADE_SCALE_IMAGE; // �������
  cascade_face_detector.detectMultiScale(ImgROI, FaceVector, searchScaleFactor, minNeighbors, flags, minFeatureSize);

  printf("��������: %d\n", FaceVector.size());
  if (FaceVector.size()>0)
  {
    Rect *object_rect = new Rect[FaceVector.size()];
    //�ֲ���Ϊȫ�ֵ�����
    for (size_t i = 0; i<FaceVector.size(); i++)
    {
      FaceVector[i].x = FaceVector[i].x + LocalArea.x;
      FaceVector[i].y = FaceVector[i].y + LocalArea.y;

      object_rect[i] = FaceVector[i];
    }

    //�Լ�⵽���������������С��������
    //ð�����򣬰���С�ķ������
    for (size_t i = 1; i <= FaceVector.size(); i++)//�������FaceVector.size()-1��
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
    
    //FaceVector��object_rectͬ���������ο�
    //���ȡǰ10������PreProcessFaceVector
    int FaceNum = 0;
    for (size_t i = 0; i < FaceVector.size(); ++i, ++FaceNum)
    {
      PreProcessFaceVector.push_back(object_rect[i]);
    }
    //printf("��������%d\n", PreProcessFaceVector.size());

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

  // ���� blob ��Ϊ����������׼����ͼƬ����ֱ�ӽ�������
  // ����Ҫ��һ�� : 1.0 / 256
  Mat inputBlob = blobFromImage(img, 1.0 / 256, Size(28, 28));

  Mat prob;
  TickMeter t;
  for (int i = 0; i < 1; i++)
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
