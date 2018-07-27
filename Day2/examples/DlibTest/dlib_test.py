import cv2
import numpy as np
import dlib

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def detect_facial_landmarks(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for index, rect in enumerate(rects):
        lm_points = predictor(gray, rect)
        lm_points = shape_to_np(lm_points)
    return lm_points

def test_01():
    img = cv2.imread('donald_trump.jpg')
    lm_points = detect_facial_landmarks(img)
    vis_img = img.copy()
    for (x,y) in lm_points:
    	cv2.circle(vis_img, (x,y), 2, (0,0,255), -1)
    cv2.imshow('', vis_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    test_01()

if __name__ == '__main__':
    main()