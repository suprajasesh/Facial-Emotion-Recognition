# Facial-Emotion-Recognition
Datasets link:

https://www.kaggle.com/datasets/deadskull7/fer2013 (csv)

https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer (images)
## Abstract
Models used- CNN, GNB, kNN, MLP, k-Means with PCA

This project depicts the performance of a set of 5 machine learning (ML) and deep learning (DL) models—CNN, GNB, kNN, MLP, mini-batch k-Means with PCA, run on FER2013
benchmark dataset and includes the inferences drawn from it for Facial Emotion classification. The FER2013 dataset is a labeled dataset for headshots depicting 
emotions in seven classes (happy, sad, angry, afraid, surprise, disgust, and neutral). The ML and DL models chosen above are used to determine the best possible model
for FER. 

## Accuracies
![image](https://github.com/suprajasesh/Facial-Emotion-Recognition/assets/112061236/e4142796-b8af-4105-bbcd-02d62c07007f)

## Inference
By analysing the performance of these algorithms on FER2013 dataset, we conclude a convolutional neural network is the best for predicting and classifying the images according to their emotions. The individual algorithms are fine-tuned for performance using different hyperparameters and performance metrics. 

Comparing the performance of MLP and CNN we see that MLP, even with accuracy boosting techniques applied, yields a much lower accuracy than CNN. This may be because of the fully connected nature of MLP wherein the number of parameters becomes unmanageable leading to redundancy and inefficiency. MLPs are also not translation invariant i.e. different angles of the same headshot may lead the model to believe that only one portion of the image may contain the eyes or certain intensity features. On the contrary, CNN considers the spatial correlation between the pixels and since layers are sparsely connected CNN architecture can be much deeper. Kernels and convolutional layers make CNN much easier to train and more efficient with image data. 

Supervised learning algorithms like kNN and GNB yield poorer results as images of FER2013 dataset belong to multiple classes, are complex as there may be overlap between the images and may lead to inefficiency as it is high-dimensional. KNN performs much better than even the unsupervised clustering (k-means). This may be because kNN is suitable for multi-class classification and the images of the test dataset are close to some instances in the training dataset hence kNN is able to compare and assign the correct label. K-means showed that we needed more clusters than the original seven. We conclude that k-means is not suitable for FER dataset as it predicts the optimal number of clusters as 256 using the pixels however there is more complex correlation between the images which can only be found using DL models. We also try to reduce the dimensionality of given FER2013 dataset using PCA and use the same for k-means clustering. The resultant PCA plot also showed k-means would not be able to find differentiable clusters in the FER dataset. 
