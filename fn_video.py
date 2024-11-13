import numpy as py
import cv2
import os.path

from fn_drawrect import draw_rect

def checkFile(myfile):
    noError = True
    if os.path.isfile(myfile) == False:
        print('File %s doesnt exist', myfile)
        noError = False
    return noError

def checkVideo(videofile):

    noError = True

    noError = checkFile(videofile)
    if noError == False:
        return noError,None

    cap = cv2.VideoCapture(videofile)
    if cap.isOpened() == False:
        print('Error opening video stream or file')
        return False,None

    return noError,cap

def sendText(image,txt,font=cv2.FONT_HERSHEY_SIMPLEX):
    (x,y) = image.shape[:2]
    #print('x is %d, y is %d', x,y)
    x1 = int(0.15*x)
    y1 = int(0.15*y)
    pos = (x1,y1)
    fontScale=0.6
    lineType = 3
    color = (255,255,0)
    cv2.putText(image,txt,pos,font,fontScale,color,lineType)
    return image

def releaseVideo(cap):	  
    cap.release()

def closeAll():	
    cv2.destroyAllWindows()

def getVideoFrames(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def isVideoOpened(cap):
    return cap.isOpened()

def cvtVideoToImages(cap,mypath='/mp4images/'):
    #total = getVideoFrames(cap)
    i = 0

    while 1:
      ret,frame = cap.read()
      i += 1
      if ret == False:
         break
      gray = cv2.cvtColor(frame,cv2.IMREAD_GRAYSCALE)
      #gray = cv2.resize(gray,(104,104))
      imgname = mypath + 'ir' +str(i) +'.jpg'
      cv2.imwrite(imgname,gray)

def parseVideoHOG(cap):
    total = getVideoFrames(cap)
    while 1:
      if cap.isOpened():
         ret,frame = cap.read()
      else:
         break
      gray = cv2.cvtColor(frame,cv2.IMREAD_GRAYSCALE)

def parseVideo_PPT(cap):

    k = 0
    while 1:
      if cap.isOpened():
         ret,frame = cap.read()
      else:
         break

      txt = "Detected"
      (x,y) = frame.shape[:2]
      print('x is %d, y is %d', x,y)
      x1 = int(0.15*x)
      y1 = int(0.15*y)
      gray = cv2.cvtColor(frame,cv2.IMREAD_GRAYSCALE)
      frame = sendText(gray,x1,y1,txt)
      if k !=32: # no space bar, draw rectangle
          frame = draw_rect(gray)
      cv2.imshow('Frame',frame)
      k = cv2.waitKey(0) & 0xff
      #print('k = ',str(k))
      if k == 27:
         break
		 
def parseVideo(cap):

    total = getVideoFrames(cap)

    curr = 0
    found = 0
    k = 0
    print('parseVideo....')
    while 1:
      if cap.isOpened():
         ret,frame = cap.read()
      else:
         break

      curr += 1
      txt = str(curr) + "/" + str(total) + " , T=" + str(found)

      #print(frame.shape[:2])
      (x,y) = frame.shape[:2]
      print('x is %d, y is %d', x,y)
      x1 = int(0.1*x)
      y1 = int(0.1*y)
      gray = cv2.cvtColor(frame,cv2.IMREAD_GRAYSCALE)
     
      frame = sendText(gray,x1,y1,txt)
      if k !=32: # no space bar, draw rectangle
          frame = draw_rect(gray)
      cv2.imshow('Frame',gray)

      #if found == 517 or found == 742 or found == 828 or found == 900 or found == 1095:
      #    cv2.imwrite('elephant_'+ str(found) + '.jpg',gray)

      k = cv2.waitKey(0) & 0xff
      #print('k = ',str(k))
      if k != 32: # no space bar
         found += 1
         
      if k == 27:
         break
      
def start(videofile='../videos/EH_1.mp4',mydst='mp4images/eh1/'):
    noError = checkFile(videofile)
    if noError:
       noError,cap = checkVideo(videofile)
    if noError:
       parseVideo_PPT(cap)
       #cvtVideoToImages(cap,mydst)
       releaseVideo(cap)
       closeAll()

#videofile = '../videos/irvideo1.mp4'
'''
dirpath='../videos/'
filepath='EH_1.mp4'
start(dirpath+filepath, 'mp4images/eh1/')
'''