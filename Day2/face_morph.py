# coding: utf-8
import cv2
import numpy as np
import dlib
import random
from pprint import pprint

#### dlib ####
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

def get_points(img):
    points = []
    lm_points = detect_facial_landmarks(img)
    for (x,y) in lm_points:
        points.append((int(x), int(y)))
    return points

#### dlib ####

#### delaunay ####
def rect_contains(rect, point) :
    # Check if a point is inside a rectangle
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def get_triangles(size, points):
    # get delaunay triangles
    # size = img.shape
    # [(x1, y1), (x2, y2), (x3, y3)]
    rect = (0, 0, size[1], size[0])

    subdiv = cv2.Subdiv2D(rect);
    '''
    points.append((0, 0))
    points.append((0, size[0]//2))
    points.append((0, size[0]))

    points.append((size[1]//2, 0))
    points.append((size[1]//2, size[0]))

    points.append((size[1], 0))
    points.append((size[1], size[0]//2))
    points.append((size[1], size[0]))
    print(points)
    ' ''
    subdiv.insert((0, 0))
    subdiv.insert((0, size[0]//2))
    subdiv.insert((0, size[0]))

    subdiv.insert((size[1]//2, 0))
    subdiv.insert((size[1]//2, size[0]))

    subdiv.insert((size[1], 0))
    subdiv.insert((size[1], size[0]//2))
    subdiv.insert((size[1], size[0]))
    '''
    #print(points)

    for p in points :
        subdiv.insert(p)
    subdiv.insert((0, 0))
    subdiv.insert((0, size[0]//2))
    subdiv.insert((0, size[0] - 1))
    subdiv.insert((size[1]//2, 0))
    subdiv.insert((size[1]//2, size[0] - 1))
    subdiv.insert((size[1] - 1, 0))
    subdiv.insert((size[1] - 1, size[0]//2))
    subdiv.insert((size[1] - 1, size[0] - 1))

    triangleList = subdiv.getTriangleList();

    triangles = []
    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3) :
            triangles.append([pt1, pt2, pt3])
        else:
            triangles.append([])
    return triangles, subdiv

def draw_voronoi(img, subdiv) :
    # Draw voronoi diagram
    (facets, centers) = subdiv.getVoronoiFacetList([])
    print(len(facets))

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        #color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        one_third_of_facets = int(len(facets) / 3) + 1
        if i < one_third_of_facets:
            color = (0, 0, (i % one_third_of_facets) * 6 + 84)
        elif i < one_third_of_facets * 2:
            color = (0, (i % one_third_of_facets) * 6 + 84, 0)
        else:
            color = ((i % one_third_of_facets) * 6 + 84, 0, 0)
        

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)

#### delaunay ####

#### face morph ####

def applyAffineTransform(src, srcTri, dstTri, size) :
    # Apply affine transform calculated using srcTri and dstTri to src and
    # output an image of size.

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

def morphTriangle(img1, img2, img, t1, t2, t, alpha) :
    # Warps and alpha blends triangular regions from img1 and img2 to img

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

#### face morph ####

if __name__ == '__main__':

    filename1 = 'hillary_clinton.jpg'
    filename2 = 'ted_cruz.jpg'
    alpha = 0.5

    # Read images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    # Read array of corresponding points
    points1 = get_points(img1)
    points2 = get_points(img2)
    points = []

    # Compute weighted average point coordinates
    #'''
    for i in range(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        points.append((int(x), int(y)))
    #'''

    '''
    # Read triangles from tri.txt
    with open("tri.txt") as file :
        for line in file :
            x, y, z = line.split()

            x = int(x)
            y = int(y)
            z = int(z)
            
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [ points[x], points[y], points[z] ]
            print(t1)

            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
    '''

    t , subdiv = get_triangles(img1.shape, points)
    t1, subdiv1 = get_triangles(img1.shape, points1)
    t2, subdiv2 = get_triangles(img1.shape, points2)

    # Allocate space for voronoi Diagram
    #img_voronoi_0 = np.zeros(img1.shape, dtype = img1.dtype)
    img_voronoi_1 = np.zeros(img1.shape, dtype = img1.dtype)
    img_voronoi_2 = np.zeros(img1.shape, dtype = img1.dtype)

    # Draw voronoi diagram
    #draw_voronoi(img_voronoi_0, subdiv)
    draw_voronoi(img_voronoi_1, subdiv1)
    draw_voronoi(img_voronoi_2, subdiv2)

    # Show results
    #cv2.imshow("img_voronoi_0", img_voronoi_0)
    #cv2.imshow("img_voronoi_1", img_voronoi_1)
    #cv2.imshow("img_voronoi_2", img_voronoi_2)
    #cv2.waitKey(0)

    '''
    # Gen t
    t = []
    for i in range(len(t1)):
        if t1[i] == [] or t2[i] == []:
            t.append([])
            continue
        trangle = []
        for j in range(3):
            x = ( 1 - alpha ) * t1[i][j][0] + alpha * t2[i][j][0]
            y = ( 1 - alpha ) * t1[i][j][1] + alpha * t2[i][j][1]
            trangle.append((x, y))
        t.append(trangle)
    '''
    
    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    len_t = min(len(t), len(t1), len(t2))
    pprint(t)
    print(len(t), len(t1), len(t2))
    for i in range(len_t):
        if t[i] != [] and t1[i] != [] and t2[i] != []:
            morphTriangle(img1, img2, imgMorph, t1[i], t2[i], t[i], alpha)

    # Display Result
    cv2.imshow("Morphed Face", np.uint8(imgMorph))
    cv2.waitKey(0)
