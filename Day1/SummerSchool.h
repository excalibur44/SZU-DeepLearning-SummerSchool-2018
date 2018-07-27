#pragma once

#ifndef _SUMMER_SCHOOL_H_
#define _SUMMER_SCHOOL_H_

#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class SummerSchool
{
private:
  Mat image;
public:
  SummerSchool() {}
  ~SummerSchool() {}
  void image_read(string filename);
  void image_reverse();
  void image_sobel();
  void video_read(string filename);
  void video_read();

  void detect_face();
};

#endif
