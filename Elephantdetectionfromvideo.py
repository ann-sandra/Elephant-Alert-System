#!/usr/bin/env python

'''
This code is duplicate copy of fn_classifiers.py stored in code directory
'''
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import os

from fn_video import checkFile, checkVideo, releaseVideo,sendText,getVideoFrames
from fn_drawrect import draw_rect
from hogsetsvm import parseSVM,loadSVM

#videoTestImageSize = (480,360)
videoTestImageSize = (200,200)

def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    samples, responses = a[:,1:], a[:,0]
    return samples, responses

def get_images(mypath,hogVal,x = 480,y=360):
    imageList=[]
    resultList=[]
    for path,dirs,files in os.walk(mypath+"\\positive\\"):
       for filename in files:
             fullpath = os.path.join(path,filename)
             print('full path filename is ', fullpath)             
             image = cv2.imread(fullpath,cv2.IMREAD_GRAYSCALE)
             #image = cv2.resize(image,(x,y))
             imageList.append(image)
             resultList.append(1)
    for path,dirs,files in os.walk(mypath+"\\negative\\"):
       for filename in files:
             fullpath = os.path.join(path,filename)
             image = cv2.imread(fullpath,cv2.IMREAD_GRAYSCALE)
             #image = cv2.resize(image,(x,y))
             imageList.append(image)
             resultList.append(0)
    samples=np.array(imageList)
    print('done get_images')
    #if hogVal == False:
    #    samples=np.float32(samples)
    #else:
    #    samples=samples.reshape(-1,104,104)
    #print('getimages: samples shape is',samples.shape)
    responses=np.array(resultList) 
    return samples,responses
    #return imageList,resultList

class LetterStatModel(object):
    class_n = 1   # Arul - old value is 26
    train_ratio = 1.0

    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses

class SVM(LetterStatModel):
    def __init__(self):
        self.model = cv2.ml.SVM_create()
        print('SVM Model initialized')

    def train(self, hogVal, samples, responses):
        #if hogVal == False:
        self.model.setKernel(cv2.ml.SVM_LINEAR)
        self.model.setType(cv2.ml.SVM_EPS_SVR) #self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setC(0.01) #(2.67)
        self.model.setGamma(0) #(5.383)
        self.model.setNu(0.5) # newly introduced as per github.com/ahmetozlu/vehicle_counting_hog_svm/blob/master/src/Main.cpp
        self.model.setP(0.1) #for EPSILON_SVR. epsilon in loss function - newly added
        self.model.setDegree(3)
        self.model.setCoef0(0.0)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT,100,1.e-03))
        
        #else:
            # self.model.setGamma(0.50625)
            # self.model.setC(12.5)
            # self.model.setKernel(cv2.ml.SVM_RBF)
            # self.model.setType(cv2.ml.SVM_C_SVC)

        #self.model.setType(cv2.ml.SVM_C_SVC)
        #self.model.setC(1)
        #self.model.setKernel(cv2.ml.SVM_RBF)
        #self.model.setGamma(.1)
        #self.model.train(samples, cv2.ml.ROW_SAMPLE, responses.astype(int))
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        print('call SVM predict')
        print('var count is ',self.model.getVarCount())
        _ret, resp = self.model.predict(samples)
        return resp.ravel()

def get_hog(x=104,y=104): 
    winSize = (x,y)
    blockSize = (8,8)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def train_samples(model,samples,responses,hogVal):
        train_n = int(len(samples)*model.train_ratio)

        print('training SVM ...')
        if hogVal == False:
            print('in false condition')
            model.train(hogVal,samples[:train_n], responses[:train_n])
        else:
            model.train(hogVal,hog_descriptors[:train_n], responses[:train_n])

        output = model.predict(samples)[1].ravel()
        print(output)

def writeToFile(fd, file,txt):
        #f = open('output.txt','w+')
        fd.write(txt+'\n')
        #f.close()
        
def predictTestImages(model,cap):
        print('Predicting test images ... new...')
        pos = (480,360)
        mypath = '../input/images/'
        outputpath='../output/'
        elephantoutputpath = outputpath + 'elephant/'
        otheroutputpath = outputpath + 'other/'
        outputfile=''
        #outputfile = '../output/output.txt'
        hog_descs=[]
        hog = get_hog()
        for path,dirs,files in os.walk(mypath):
            for filename in files:
               fullpath = os.path.join(path,filename)
               gray = cv2.imread(fullpath,cv2.IMREAD_GRAYSCALE)
               gray = cv2.resize(gray,pos)
               #outputfile = outputpath + filename
               #cv2.imwrite(outputfile,gray)
               #gray = cv2.imread(outputfile,cv2.IMREAD_GRAYSCALE)					
               hog_desc = hog.compute(gray)
               hog_desc = np.float32(hog_desc)
               result = model.predict(np.ravel(hog_desc)[None,:])
               #print(result)
               result = result[1].ravel()
               flag = False
               if result[0] == 1.0:
                   flag = True
                   outputfile = elephantoutputpath + filename
                   print(outputfile + ' is true')
               else:
                   outputfile = otheroutputpath + filename
                   print(outputfile + ' is false')
               if flag == True:
                   gray = draw_rect(gray)
               cv2.imshow('final',gray)
               cv2.imwrite(outputfile,gray)
               k = cv2.waitKey(0) & 0xff
               if k == 27:
                   break

def predictBulkSamples(model,samples,responses,hogVal):
        print('Defining HoG parameters ...')
        # HoG feature descriptor
        hog = get_hog(104,104);

        print('testing...')
        print('Predicting the image ... ')
        hog_descriptors = []
        for img in samples:
             hog_descriptors.append(hog.compute(img))
        hog_descriptors = np.squeeze(hog_descriptors)
        train_n = int(len(hog_descriptors)*0.9)
        print('total is %d train_n is %d',int(len(hog_descriptors)),train_n)
        #test_rate  = np.mean(model.predict(hog_descriptors[:train_n]) == responses[:train_n].astype(int))
        #print('test_rate is %f ' % (test_rate*100))
        result = model.predict(hog_descriptors[train_n:])
        print(result)
        result = result[1].ravel()
        print(result)
        err = (result != responses)
        print(err)
        err = err.mean()
        print('error: %.2f %%' % (err*100))		

if __name__ == '__main__':
    import getopt
    import sys

    args, dummy = getopt.getopt(sys.argv[1:], '', ['model=', 'data=', 'load=', 'save=','test='])
    args = dict(args)
    args.setdefault('--model', 'svm')
    #args.setdefault('--data', '../images/elephant')
    #args.setdefault('--test', videofile)

    hogVal = False
    print('main program...')
    
    if '--data' in args:
        print('image path is %s' % args['--data'])
        samples,responses = get_images(args['--data'],hogVal,104,104)
  
    cap = None

    if '--test' in args:
        print('test..args')
        fn = 'elephant480360.xml'
        #model = None
        model = cv2.ml.SVM_load(fn)
        print('call predicttest')
        predictTestImages(model,cap)

    if '--save' in args:
        model = SVM()
        train_samples(model, samples,responses,hogVal)
        fn = args['--save']
        print('saving model to %s ...' % fn)
        model.save(fn)

    if '--load' in args:
        fn = args['--load']
        print('loading model from %s ...' % fn)
        #model.load(fn)
        model = cv2.ml.SVM_load(fn)
        if '--data' in args:
            predictBulkSamples(model,samples,responses,hogVal)

    cv2.destroyAllWindows()

'''
In order the run the program give the command as:
$python elephant_video_code.py --video=..\\videos\\e3.mp4

NOTE 1: don't give quotes for the file
NOTE 2: Press Esc to exit, while running the video

to test the iamges...
1. put the images in input/images folder 
2. output can be seen in predictTestImages code .. in the output folder output/elephant or output/other
3. To run the code
python elephant_video_code.py --test=test

'''
