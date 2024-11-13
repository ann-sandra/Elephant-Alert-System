import numpy as np
import pickle
import re
import cv2
import xml.etree.ElementTree as ET

def parseSVM(myfile,storefile):
   #svm = cv2.ml.SVM_load(myfile)
   tree = ET.parse(myfile)
   root = tree.getroot()
   SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
   rho = float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text)
   svmvec = [float(x) for x in re.sub('\s+',' ',SVs.text).strip().split(' ')]
   svmvec.append(-rho)
   pickle.dump(svmvec,open(storefile,'wb'))

def loadSVM(img,storefile):
   pos = (480,360)
   hog = cv2.HOGDescriptor(pos,(4,4),(8,8),(8,8),9)
   svm = pickle.load(open(storefile,'rb'))
   hog.setSVMDetector(np.array(svm))
   image = cv2.imread(img,0)
   image = cv2.resize(image,pos)
   meanShift=False
   (rects,weights) = hog.detectMultiScale(image,winStride=(4,4),padding=(16,16),scale=1.05,useMeanshiftGrouping=meanShift)
   print(rects, weights)
   flag = False
   for (x,y,w,h) in rects:
      flag = True
      cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
   #cv2.imshow('Detections',image)
   #cv2.waitKey(0)
   return image,flag
   
def loadDefaultSVM(img,storefile):
   pos = (480,360)
   hog = cv2.HOGDescriptor()
   hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
   image = cv2.resize(image,pos)
   meanShift=False
   (rects,weights) = hog.detectMultiScale(image,winStride=(4,4),padding=(16,16),scale=1.05,useMeanshiftGrouping=meanShift)
   print(rects)
   for (x,y,w,h) in rects:
      cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
   cv2.imshow('Detections',image)
   cv2.waitKey(0)
   
   
#parseSVM('./tiger_v4.xml','save.p')
#img = '../sample/person.jpg'
#loadSVM(img,'save.p')
