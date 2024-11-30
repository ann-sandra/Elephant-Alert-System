# **Detection of Wild Elephants Using Image Processing on Raspberry Pi**

## **Introduction**
Elephant populations face a shortage of resources like food and water, pushing them into human habitats and escalating human-elephant conflict. Addressing this issue is a growing research area. This project employs image processing on a Raspberry Pi, combined with a Convolutional Neural Network (CNN), to detect elephants accessing forest borders.

The project demonstrates the feasibility of deploying a low-power, cost-efficient Raspberry Pi for real-time edge computing. It utilizes feature detection techniques (SIFT, SURF, HOG, and ORB) to identify the most reliable method for detecting elephants with high accuracy.

---

## **Dataset**
- **Input Directories:**
  - `input/images/elephant` - Contains elephant images.
  - `input/images/other` - Contains images of non-elephant objects.

---

## **Software Requirements**
- **Operating System:** Raspbian  
- **IDE:** Thonny  
- **Machine Learning Framework:** TensorFlow (1.14.0), Keras-preprocessing (1.1.0)  
- **Preprocessing Module:** OpenCV  

---

## **Data Preprocessing**
1. **Image Loading:** Input images are converted to grayscale.
2. **Feature Scaling:** Pixel values are normalized by dividing by 255, converting them to a range of [0, 1].

---

## **Feature Detection Techniques**
### 1. **SIFT (Scale-Invariant Feature Transform)**
   - Provides scale and rotation invariant features for robust object detection.
   - Achieved the **highest accuracy of 85%** when paired with CNN.

### 2. **SURF (Speed-Up Robust Features)**
   - Optimizes SIFT by approximating the Laplacian of Gaussian with a box filter, making it computationally faster.
   - Accuracy: **75%**.

### 3. **HOG (Histogram of Oriented Gradients)**
   - Extracts global features by analyzing gradients in small image cells and normalizing them.
   - Used in feature descriptor calculations.

### 4. **ORB (Oriented FAST and Rotated BRIEF)**
   - A fast binary descriptor resistant to noise, optimized for real-time applications.
   - Accuracy: **75%**.

---

## **Model Building**
### **CNN Architecture**
1. Input image dimension: **64×64**
2. **Convolutional Layer:** 32 filters (3×3 size)
3. **Pooling Layer:** 2×2 size  
4. Repeated convolution and pooling layers three more times.
5. **Dense Layer:** Flattens the output to a single dimension.
6. **Final Dense Layer:** Classifies images (elephant or other).

---

## **Utility Functions**
1. **`load_base(fn)`**: Loads dataset from a folder.
2. **`get_images(mypath, hogVal, x, y)`**: Reads and prepares grayscale images for training.

---

## **Feature Extraction**
- **`get_hog(x, y)`**: Configures HOG descriptors for feature extraction.

---

## **Training and Evaluation**
1. **`train_samples`**: Trains the CNN on extracted features.  
2. **`predictBulkSamples`**: Tests the model on a dataset and evaluates performance.  

- **Accuracy Scores:**
  - SIFT-CNN: **85%**
  - SURF-CNN: **75%**
  - ORB-CNN: **75%**

### **Performance Evaluation**
- **Confusion Matrix:** Used to calculate precision, recall, F1-score, and support.
- **ROC Curve:** Visualizes the trade-off between true positive rate (TPR) and false positive rate (FPR).
  - SIFT-CNN Sensitivity: **0.85**
  - SURF and ORB Sensitivity: **0.75**

---

## **Prediction Functions**
### **Image Prediction**
- **`predictTestImages`**: Classifies input images as "elephant" or "other" and saves the results in respective folders.

### **Video Prediction**
1. **`predictTestVideoSamples`**  
   - Converts video frames into grayscale and predicts elephant presence for each frame.
2. **`predictVideoSamples`**  
   - Processes video samples in real time and visualizes results with bounding boxes.

### **Output Logging**
- **`writeToFile`**: Writes detection results to a file, which can trigger an alarm if an elephant is detected.

---

## **Visualization**
- **`draw_rect(image)`**: Draws bounding boxes around detected elephants in images or video frames.

---

## **Key Findings**
1. SIFT-CNN demonstrated the highest accuracy (**85%**) for elephant detection.
2. Low-power Raspberry Pi is effective for real-time edge computing.
3. Robust feature extraction techniques like SIFT and HOG enhance detection capabilities.

---

## **Project Directory Structure**
```plaintext
project/
│
├── input/
│   ├── images/
│   │   ├── elephant/
│   │   └── other/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   ├── model.py
│
├── output/
│   ├── predictions/
│   ├── logs/
│
└── README.md
```
---

## **Conclusion**
This project successfully demonstrates a scalable, low-cost system for detecting wild elephants using machine learning on Raspberry Pi. The SIFT-CNN combination was identified as the most effective approach, achieving an accuracy of 85%. This solution can be easily deployed for edge computing to help mitigate human-elephant conflict.
