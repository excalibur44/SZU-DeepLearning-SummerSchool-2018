// SummerSchool.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "SummerSchool.h"

int main()
{
  SummerSchool ss;
  string image_name = "res/face_test.jpg"; // or res/boldt.jpg

  /* MISSION 1 */
  ss.image_read(image_name);
  
  /* MISSION 2 */
  ss.image_reverse();

  /* MISSION 3 */
  ss.image_sobel();

  /* MISSION 4 */
  ss.detect_face();

  /* OTHER */
  //ss.video_read("res/face_test.avi");

	return 0;
}

