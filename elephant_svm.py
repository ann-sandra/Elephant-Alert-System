# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:09:03 2018
@author: Arulg

This code is taken from the folder code/cameraTrap/HogSvm2.py which is main file
"""

import cv2
import os
import numpy as np
from shutil import move

trainingImages = []
trainingLabels=[]
testingImages = []
testingLabels=[]
savefile = ".\\" + 'svm_data_elephant200200.xml'

train_dir = "..\\input\\train\\"
test_dir = "..\\input\\test\\"
realTest_dir="..\\input\\realtest\\"

#imageSize=(480,360)
imageSize=(200,200)
    
svm=cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

def addToTrainingList(mydir, label):
    #imageSize=(200,200)
    hog = cv2.HOGDescriptor()
    file_name = ""
    for filename in os.listdir(mydir):
        if filename.endswith(".jpg"):
            file_name = os.path.join(mydir,filename)
            image=cv2.imread(file_name,0)
            #print(file_name)
            image.resize(imageSize)#Resize all images to be of the same size
            h = hog.compute(image)
            trainingImages.append(h) #Flatten image and save
            trainingLabels.append(label)

def addToTestingList(mydir, label):
    hog = cv2.HOGDescriptor()
    file_name = ""
    for filename in os.listdir(mydir):
        if filename.endswith(".jpg"):
            file_name = os.path.join(mydir,filename)
            image=cv2.imread(file_name,0)
            #print(file_name)
            image.resize(imageSize)#Resize all images to be of the same size
            h = hog.compute(image)
            testingImages.append(h) #Flatten image and save
            testingLabels.append(label)
            
def doTrain(saveFlag):
    #myfile="."
    print('start training....')
    #Create an array where each row is a flattened training image
    directory = train_dir + "\\positive"
    addToTrainingList(directory,1)
    directory = train_dir + "\\negative\\cats"
    addToTrainingList(directory,0)
    directory = train_dir + "\\negative\\dogs"
    addToTrainingList(directory,0)
    directory = train_dir + "\\negative\\forest"
    addToTrainingList(directory,0)
    
    #print(listing)
    training_data=np.array(trainingImages)
    #print(training_data)
    #print(training_label)
    
    #Only integers can be given as input
    training_data=np.float32(training_data)
    responses=np.array(trainingLabels)
    print(training_data.shape)
    #print(responses)

    if saveFlag == True:
        print('Save to file....')

    svm.train(training_data, cv2.ml.ROW_SAMPLE, responses)
    
    if saveFlag == True:    
        svm.save(savefile)
        print('File saved....end')

def doRealTest(saveFlag):
    directory = realTest_dir
    addToTestingList(directory,1) # adding dummy labels
#    directory = realTest_dir + "negative"
#    addToTestingList(directory,0)
    #print(listing)
    testing_data=np.array(testingImages)
    #print(testing_data)
    #print(testing_label)
    
    #testData=np.float32(testing_data).reshape(-1,40000)
    print('testing....')
    #see the difference...the below line is not in training......otherwise error in predict
    #testData=np.float32(testing_data)
    testData=testing_data
    print(testData.shape)

    if saveFlag == True:
        print('loading file....')
        svm.load(savefile)

    print('.predicting....')
    result = svm.predict(testData)
    
    print('print result')
    print(result)
    showResult(result)
		
def doTest(saveFlag):
    directory = test_dir + "positive"
    addToTestingList(directory,1)
    directory = test_dir + "negative\\cats"
    addToTestingList(directory,0)
    directory = test_dir + "negative\\dogs"
    addToTestingList(directory,0)
    directory = test_dir + "negative\\forest"
    addToTestingList(directory,0)    

    #print(listing)
    testing_data=np.array(testingImages)
    #print(testing_data)
    #print(testing_label)
    
    #testData=np.float32(testing_data).reshape(-1,40000)
    print('testing....')
    #see the difference...the below line is not in training......otherwise error in predict
    #testData=np.float32(testing_data)
    testData=testing_data
    print(testData.shape)

    if saveFlag == True:
        print('loading file....')
        svm.load(savefile)

    print('.predicting....')
    result = svm.predict(testData)
    
    print('print result')
    print(result)
    showAccuracyResult(result)

def showAccuracyResult(result):
    print('show accuracy result')
    correct=0
    wrong=0
    for i in range(0,len(testingLabels)):
        if(result[1][i] == testingLabels[i]):
            print("Index ",i," true; val is ",result[1][i])
            correct+=1
        else:
            print("Index ",i," false", result[1][i])                        
            #print("Predicted = %d" % result[1][i])
            #print("Actual = %d" % testingLabels[i])
            wrong+=1
    print(correct)
    print(wrong)

    #######   Check Accuracy   ########################
    accuracy=(correct)/(correct+wrong)*100
    print(accuracy)
    
def showResult(result):
    print('show result')
    #for i in range(0,len(testingLabels)):
    dir_dst_pos = realTest_dir + "positive\\"
    dir_dst_neg = realTest_dir + "negative\\"
    
    i = 0
    for file in os.listdir(realTest_dir):
        if file.endswith(".jpg"):
            src_file = os.path.join(realTest_dir, file)

            if(result[1][i]):
                print("Index ",i," :  val is ",result[1][i])
                dst_file = os.path.join(dir_dst_pos, file)
            else:
                print("Index ",i," : val is", result[1][i])            
                dst_file = os.path.join(dir_dst_neg, file)
            ++i
            shutil.move(src_file,dst_file)

'''    
doTrain(False)
#doTest(False)
doRealTest(False)
'''
doTrain(True)