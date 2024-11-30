# **Detection of Wild Elephants Using Image Processing on Raspberry Pi**

## **Introduction**
Elephant populations are increasingly encroaching into human habitats due to dwindling natural resources like food and water, resulting in severe human-elephant conflicts. To address this issue, this project implements an automated elephant detection system using image processing techniques on a Raspberry Pi. By integrating feature detection methods with a Convolutional Neural Network (CNN), the project provides a scalable, low-cost solution for real-time monitoring of elephants at forest borders.

The system's primary goal is to detect elephants accurately and trigger necessary alerts to prevent conflicts, thus preserving both human safety and wildlife.

---

## **Project Objectives**
1. To detect elephants at forest borders using Raspberry Pi.
2. To integrate multiple feature detection techniques and evaluate their effectiveness.
3. To develop a robust machine learning model (CNN) for real-time image classification.
4. To implement a low-power edge computing system for easy deployment.
5. To evaluate and compare the performance of different feature extraction methods (SIFT, SURF, HOG, ORB).

---

## **Key Features**
- **Real-time Detection:** Efficient edge computing for real-time monitoring and predictions.
- **Feature Detection Techniques:** Comparative analysis of SIFT, SURF, HOG, and ORB.
- **Energy-Efficient Deployment:** Low power consumption using Raspberry Pi.
- **Scalable and Cost-Effective:** Ideal for deployment in remote areas with limited resources.

---

## **Dataset**
The dataset contains images categorized into two classes:
- **Elephant:** Images of elephants captured from open-source repositories.
- **Other:** Images of non-elephant objects or backgrounds for model training.

### **Dataset Structure**
```plaintext
dataset/
│
├── input/
    ├── images/
        ├── elephant/
        └── other/

```


### **Input Directories**
- `input/images/elephant`: Contains elephant images.
- `input/images/other`: Contains non-elephant images.

---

## **Software Requirements**
- **Operating System:** Raspbian  
- **IDE:** Thonny  
- **Machine Learning Framework:** TensorFlow (1.14.0), Keras-preprocessing (1.1.0)  
- **Preprocessing Module:** OpenCV  

---

## **Hardware Requirements**
- **Device:** Raspberry Pi 4 Model B
- **Camera Module:** Raspberry Pi Camera Module v2
- **Power Supply:** 5V, 3A power adapter
- **Storage:** Minimum 32GB SD card for datasets and model storage

---

## **Data Preprocessing**
1. **Image Loading:** 
   - Images are read and converted to grayscale for simplicity.
2. **Feature Scaling:** 
   - Pixel values are normalized by dividing by 255 to scale them to a range of [0, 1].
3. **Data Augmentation:** 
   - Techniques like flipping, rotation, and cropping are applied to expand the dataset.

---

## **Utility Functions**
1. **`load_base(fn)`**: Loads the dataset from the specified folder.
2. **`get_images(mypath, hogVal, x, y)`**: Reads and prepares grayscale images for training.
3. **`draw_rect(image)`**: Draws bounding boxes around detected features for visualization.
4. **`writeToFile(fd, file, txt)`**: Logs detection results to a file for further processing.

---

## **Feature Detection Techniques**
### 1. **SIFT (Scale-Invariant Feature Transform)**
   - Extracts scale and rotation invariant features.
   - Most reliable method with **85% accuracy** when paired with CNN.

### 2. **SURF (Speed-Up Robust Features)**
   - Optimized SIFT using box filters for faster computation.
   - Accuracy: **75%**.

### 3. **HOG (Histogram of Oriented Gradients)**
   - Decomposes images into cells, computes gradients, and normalizes for feature extraction.

### 4. **ORB (Oriented FAST and Rotated BRIEF)**
   - A binary descriptor for fast and robust object detection.
   - Accuracy: **75%**.

---

## **Model Architecture**
### **Convolutional Neural Network (CNN)**
1. **Input Layer:**  
   - Accepts images with dimensions **64×64**.
2. **Convolutional Layers:**  
   - Uses **32 filters** of size **3×3** in the first layer.
   - Repeated convolutional and pooling steps three more times.
3. **Pooling Layers:**  
   - Max pooling with a filter size of **2×2**.
4. **Dense Layer:**  
   - Flattens the final output and connects to a fully connected layer.
5. **Output Layer:**  
   - Binary classification (elephant or not).

### **Model Evaluation**
- **Metrics:** Precision, Recall, F1-Score, Accuracy
- ### Results Overview

    #### SIFT
    - **Accuracy Score:** 0.85
    - **Metrics:**
      - Precision: 0.85
      - Recall: 0.85
      - F1-Score: 0.85

    #### SURF
    - **Accuracy Score:** 0.75
    - **Metrics:**
      - Precision: 0.75
      - Recall: 0.75
      - F1-Score: 0.75
    
    #### ORB
    - **Accuracy Score:** 0.75
    - **Metrics:**
      - Precision: 0.75
      - Recall: 0.75
      - F1-Score: 0.75


- **SIFT** performed the best with an accuracy of **0.85**.
- **SURF** and **ORB** both achieved an accuracy of **0.75**, with similar metric scores.

- **ROC Curve:** Used to visualize model performance with a threshold of:
  - SIFT: **0.85 Sensitivity**
  - SURF and ORB: **0.75 Sensitivity**

---

## **Training and Testing**
### **Training Pipeline**
1. Extract features using the selected method (SIFT, SURF, HOG, or ORB).
2. Train the CNN on the preprocessed dataset.
3. Evaluate the model using test images.

### **Testing and Predictions**
1. **Image Prediction:** 
   - Classifies test images into "elephant" or "other".
   - Saves results in output folders.
2. **Video Prediction:** 
   - Processes video frames and predicts elephant presence.
   - Visualizes results with bounding boxes for detected elephants.

### **Output Logging**
- **Real-Time Alarm System:** Triggers an alarm if an elephant is detected based on log files.
- **Confusion Matrix:** Used to calculate and log accuracy, precision, recall, and F1-score.

---

## **Performance Comparison**
| **Feature Detection Method** | **Accuracy** | **Sensitivity** | 
|------------------------------|--------------|-----------------|
| SIFT + CNN                   | 85%          | 0.85            | 
| SURF + CNN                   | 75%          | 0.75            | 
| ORB + CNN                    | 75%          | 0.75            | 

---

## **Visualization**
### **ROC Curve:**
- Receiver Operating Characteristic plot visualizes TPR vs. FPR for binary classification.
- Threshold:
  - **SIFT-CNN:** 0.85
  - **SURF/ORB-CNN:** 0.75

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

## **Key Findings**
1. SIFT-CNN demonstrated the highest accuracy (**85%**) for elephant detection.
2. Low-power Raspberry Pi is effective for real-time edge computing.
3. Robust feature extraction techniques like SIFT and HOG enhance detection capabilities.

---

## **Future Enhancements**
1. Improve model accuracy using more advanced neural network architectures.
2. Expand the dataset with additional elephant and non-elephant images.
3. Implement a mobile application for real-time alerts and monitoring.
4. Integrate IoT-based alarm systems for remote areas.

---

## **Conclusion**
This project successfully showcases a scalable, low-cost solution for detecting wild elephants using Raspberry Pi and CNN. The SIFT-CNN model emerged as the most effective with an accuracy of 85%, making it a reliable choice for real-time edge computing applications. This system can play a crucial role in mitigating human-elephant conflict and promoting harmonious coexistence.
