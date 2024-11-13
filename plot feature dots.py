import numpy as np
import cv2
import os.path

def draw_rect(image):
   #find the size of the image
   (y,x) = image.shape[:2]
   #print('shape is ',x,y)

   x1 = int(0.1*x)
   y1 = int(0.1*y)
   x2 = x - x1
   y2 = y - y1
   x3 = int(0.05*x)
   y3 = int(0.05*y)
   font = cv2.FONT_HERSHEY_SIMPLEX
   cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,255),2)
   #cv2.putText(im,'Tiger', (x3,y3),font,0.5,(11,255,255),2,cv2.LINE_AA)
  
   return image

def rect_show():
   myfile = '../sample/tiger.jpg'
   if os.path.isfile(myfile) == False:
      print('File %s not found',myfile)
      exit()
   else:
      im = cv2.imread(myfile)
      im = draw_rect(im)
      cv2.imshow('final',im)
      cv2.waitKey(0)
