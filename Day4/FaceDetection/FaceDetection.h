#pragma once

#ifndef _FACE_DETECTION_H_
#define _FACE_DETECTION_H_

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
using namespace std;

/* ==== function ==== */
void initialize();
void detect_face(Mat image, Rect LocalArea, vector<Rect> &FaceVector, vector<Rect> &PreProcessFaceVector);
void recognize_face(Mat img, int *classId, double *classProb);

/* ==== metadata ==== */
static String imageFile = "res/0004.jpg";
static String modelTxt = "res/deploy.prototxt";
static String modelBin = "res/lenet_iter_6000.caffemodel";
static String cascade_path = "res/haarcascade_frontalface_alt.xml";
static char* labels_filename = "res/labels.txt";

/* ==== global variable ==== */
CascadeClassifier  cascade_face_detector;
vector<String> classNames;
Net net;

#endif
