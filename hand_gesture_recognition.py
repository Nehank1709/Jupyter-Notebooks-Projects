import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)   #instantiates an object for videoCapture from web cap
while (cap.isOpened()):

  #read image
  ret, img = cap.read()  # store the frame in img variable, ret stores whether frame successfullly read or not

  #get hand data from rectangle sub window on the screen
  cv2.rectangle(img, (300,300), (100, 100), (0, 255, 0), 0)  #if hand comes to this rectangle it will be detected
  crop_img = img[100:300, 100:300]

  #convert to grayscale 
  grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)   #and then to binary to find our region of interest

  #applying gaussian blur
  value = (35, 35)
  blurred = cv2.GaussianBlur(grey, value, 0)

  # thresholding: Otsu's Binarization method
  _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)    # low pass filter that allows only particular color
  # ranges to be highlighted as white and rest colors are suppressed by showing them in black
  # using otsu opencv automatically calculates and approximates the threshold value of a bimodal image from its image histogram
  # pixels greater than 127 assigned 1 and pixels lesser than 255 assigned 0

  # show threshold image
  cv2.imshow('Threshold', thresh1)

  #check Opencv version to avoid unpacking error
  (version, _, _) = cv2.__version__.split('.')

  if version == '3':
    image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  elif version == '4':
    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#find contour with max area
  cnt = max(contours, key = lambda x : cv2.contourArea(x))

#create bounding rectangle around the contour (can skip below two lines)
  x, y, w, h = cv2.boundingRect(cnt)
  cv2.rectangle(crop_img, (x,y), (x+w, y+h), (0, 0, 255), 0)

#finding convex hell
  hull = cv2.convexHull(cnt)

# drawing contours
  drawing = np.zeros(crop_img.shape, np.uint8)
  cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
  cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

#finding convex hull
  hull = cv2.convexHull(cnt, returnPoints=False)

# finding convexity defects
# assumption is that any defect in convex hull is due to fingers
  defects = cv2.convexityDefects(cnt, hull)
  count_defects = 0
  cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)  #draw contours on above defects

#applying Cosine rule to find angle for all defects (between fingers)
# with angle > 90 degrees and ignore defects

  for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]

    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])

# find length of all sides of triangle
# of the particular defect
    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

  #apply cosine rule here
    angle = math.acos((b **2 + c**2 - a**2)/ (2*b*c)) * 57

  # ignore angle greater than 90 and highlight rest with red dots
    if angle <= 90:
      count_defects += 1
      cv2.circle(crop_img, far, 1, [0, 0, 255], -1)

  #dist = cv2.pointPolygonTest(cnt, far, True)

  # draw a line from start to end i.e. the convex points (finger tips)
  # (can skip this part)
    cv2.line(crop_img, start, end, [0, 255, 0], 2)
  #cv2.circle(crop_img, far, 5, [0, 0, 255], -1)

#define actions required
  if count_defects == 1:
    cv2.putText(img, "Detected 2 fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

  elif count_defects == 2:
    cv2.putText(img, "Detected 3 fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    
  elif count_defects == 0:
    cv2.putText(img, "Detected 1 finger", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

  elif count_defects == 3:
    cv2.putText(img, "Detected 4 fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

#   elif count_defects == 4:
#     cv2.putText(img, "Detected 4 fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

  else:
    cv2.putText(img, "An Entire Hand", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)


# show appropriate images in windows
  cv2.imshow('Gesture', img)
  all_img = np.hstack((drawing, crop_img))
  cv2.imshow('Contours', all_img)

  k = cv2.waitKey(10)
  if k == 27:   # if esc pressed break
    break
