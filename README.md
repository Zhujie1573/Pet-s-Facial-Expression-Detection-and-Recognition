# Pet's Facial Expression Detection and Recognition
![My Pet's Image](repo_img/happy_animal.png)
## Background
As computer vision techniques excel in human-centered fields, their adaptation for pet-centric applications becomes increasingly prevalent, deepening our insights into pets' emotions and mental health. Building on this momentum, several studies have explored this realm. Boneh-Shitrit et al.<sup>[1](#ref1)</sup> experimented with various deep-learning architectures to classify dogs' emotions based on their pictures. Sinnott et al.<sup>[2](#ref2)</sup> used a more extensive dataset, including 52 dog species and 23 cat species, and 300 images per species. They developed a machine learning model to classify both the pet's species and their associated facial expressions. The novel Convolutional Neural Network model IWOA–CNN designed by Yan et al.<sup>[3](#ref3)</sup> demonstrated its efficiency in dog expression recognition. However, further studies are still needed. In this project, we will focus on using computer vision techniques to detect and classify facial expressions on more diverse pets on a dataset containing 6+ pet categories (dogs, cats, rabbits, goats, etc.) and 4 facial expressions (happy, angry, sad, and other).

## Dataset
Our dataset comprises 1,000 jpg images of pet faces, classified into four emotional expressions: Sad, Angry, Happy, and Other (with each category precisely containing 250 images). The images showcase more than six types of pets to ensure diversity and inclusivity of the trained model. As the images are classified by animal behavior experts, the labels are considered sufficiently accurate to train our models. The dataset can be found in [our repo](https://github.com/Zhujie1573/CS4641-Team12/tree/main/dataset), or through the [link](https://drive.google.com/file/d/1ULijujD0HWwX2qqcKBQPOiZqBs5YR86o/view?usp=sharing).
<br>
<img src="repo_img/datadetail.png" width="250" height="400"><br>

## Problem Definition & Motivation
Pets are our most loyal companions, seamlessly blending into our families. They stand by us in moments of joy, celebrating life's highs, and offer solace during our lowest ebbs. What sets them apart is their remarkable ability to discern our emotions and respond with unwavering support. However, when it comes to reciprocating this understanding of their emotions, the task becomes less straightforward. Accurately gauging our pets' feelings is a crucial endeavor with profound implications for their mental well-being. Regrettably, many pet owners lack the necessary experience and expertise to decode their furry friends' emotions. To bridge this gap and provide the best care possible for our beloved companions, we envision creating a sophisticated machine learning model. This innovative tool will empower pet owners with the ability to recognize and understand their pets' emotions, thereby allowing them to better attend to their pets' mental health needs. After all, our pets are not just our closest friends; they are cherished members of our families deserving of our utmost care and attention.

## Data Preproccesing
### 1. PCA
1. Effect: PCA Principal Component Analysis) can reduce dimensionality, reduce noise, and increase the computation efficiency which can increase the training speed.
2. Why PCA? The image data we are dealing with has high dimension. PCA can help reduce the dimensionality and prevent the overfitting issue. The feature extraction also helps the model have better generalizability.
3. How we did PCA: We implemented reduce the dimensionality of the images via PCA through SVD. While reducing the dimensionality of an image can prevent overfitting and save some space, we want to first make sure we compress the image to the appropriate number of components. The 'number of components' vs 'variance' plot is generated. In the DenseNet-121 we choose to use num_components=89 since its corresponding variance is very close 1. Using the code we wrote in HW3, we generated the plot for each number of component on the x-axis to confirm our idea. It's clear that very small number of components will result losing important information and num of components larger than 67 look basically the same. Using PCA or not and finding the correct number of components to compress the image are importnat question to answer. We will continue to tweak this parameter.
<p align="center">
  <img src="repo_img/component_variance_pca.png" width="600" height="300"><br>
  <img src="repo_img/component_gird_pca.png" width="600" height="600">
</p>

## Method & Algorithm
In our project, we'll employ advanced deep learning methods and algorithms. We'll use Keras, a high-level neural networks API, to create a Convolutional Neural Network (CNN) using a Sequential model. We'll first tap into existing popular classification models like ResNet50 <sup>[4](#ref4)</sup>, DenseNet121 <sup>[5](#ref5)</sup>, and EfficientNetB0, using TensorFlow's Keras applications. To further enhance our model's performance, we will select the model with best performance and modify or add essential layers. Our CNN will include layers such as Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, and BatchNormalization. Additionally, we'll apply regularization techniques to prevent overfitting. This approach leverages state-of-the-art deep learning techniques to achieve our project's goals effectively.<br>

In evaluating our model, we'll utilize accuracy for a holistic assessment of classification across all facial expression categories, and the F1-score will elucidate the trade-off between precision and recall. Moreover, to address potential class imbalances, the AUC-ROC will be used to gauge discriminative performance. Lastly, a confusion matrix will be used to pinpoint specific misclassification instances.

### 1. EfficientNet
#### a. Introduction and Advantages
EfficientNets are advanced neural networks developed by Google that are designed to scale up more efficiently, meaning they can get better accuracy with less computational cost. They do this by adjusting network depth, width, and image resolution based on a set formula, which is different from just increasing the size of the network randomly. EfficientNets balance high accuracy and efficiency. It performs very well on image recognition tasks and doesn't need as much computational power as other models.
#### b. Implementation
In our project, we used a version of EfficientNet called B5. It's pre-trained, meaning it has already learned from a huge dataset called ImageNet. We made it fit our needs by changing the input shape to match our images and by adding layers that make decisions based on the number of classes we have. We kept the original learned patterns intact and added some techniques like batch normalization and dropout to make the model more reliable and to prevent it from just memorizing the training data. We used an optimizer called Adamax and a loss function suited for classifying images into categories. We also set up the training to stop automatically if the model isn't improving, to save time and resources.
#### c. Result
<p align="center">
<img src="repo_img/effacc.png" width="350" height="350"><br>
</p>
<p align="center">
<img src="repo_img/effloss.png" width="350" height="350"><br>
</p>
<p align="center">
<img src="repo_img/effMatrix.png" width="350" height="350"><br>
</p>
The results from my EfficientNet model illustrate my learning journey through accuracy and loss metrics over 100 epochs, as well as a confusion matrix evaluating my classification performance. In the first graph, my training accuracy (red line) and validation accuracy (green line) increase over time, which shows that I am learning effectively. I reached my best accuracy at epoch 83. In the second graph, both training and validation losses decrease as I learn, with my best loss marked at epoch 89, indicating that my predictions are becoming more precise. The confusion matrix reveals my performance on a multi-class classification task with labels like "Angry", "Sad", "Happy", and "Other". It shows that I am quite proficient at classifying "Sad" and "Happy" emotions, although there are some misclassifications, mainly between "Sad" and "Other". Overall, I am on a positive learning curve and demonstrate decent classification capabilities, with opportunities for improvement in differentiating certain classes.

#### d. Discussion and Future Work
One issue with EfficientNet is that it still needs a lot of data and computing power to start with. While its method for increasing size is systematic, it might not be perfect for all types of data or tasks. To deal with this, we could tweak the model more specifically for our data or use other methods to optimize the network's structure.

### 2. DenseNet-121 with PCA
#### a. Introduction and Advantages
DenseNet-121 introduces a unique architectural innovation in deep learning with its dense connectivity pattern, where each layer connects to every other layer in a feed-forward fashion. This design not only streamlines the training process by enhancing feature propagation but also reduces the model's complexity by minimizing the number of parameters. Such characteristics make DenseNet-121 exceptionally efficient for image classification tasks, capturing the attention of computer science researchers for its elegant handling of the vanishing-gradient problem and feature redundancy.
#### b. Implementation
This Kaggle dataset already helped us partition the image into training, validation, and test datasets. All the input data underwent PCA compression with number of components=89. The DensenNet-121 is trained 100 epochs in each experiment and validate every epoch.
* Experiment 1: We first experimented with directly use DenseNet-121 with PCA preprocessing to train the pet emotion classification problem from scratch. The experiment results are not good. The accuracy of the model on the test dataset is 52.63%, meaning the model is slightly better than random guess (since we have 4 emotion categories, the random guess will result in a 25% accuracy on average). Therefore, we decided to switch to pretrained DenseNet-121 and do transfer learning on our pets emotion datasets.
* Experiment 2: The DenseNet-121 we used is pretrained on ImageNet. In order to compare the model performace across different architectures, similar hyperparameters of EfficientNet was applied to DenseNet-121, where batch normalization, dropout layer with p=0.45, and Adamax optimizer were used. Specifically for the DenseNet-121, we incorperated dense layers with rectified linear unit (ReLU) activations that incorporate L1 and L2 regularization to further reduce overfitting.
#### c. Result
The model can classify pet's emotion correctly in most of the time. Here are two examples of the model's classification.
<p align="center">
<img src="repo_img/dense_train_valid.png" width="750" height="350"><br>
</p>
During the testing phase, DenseNet-121 in Experiment 2 can correctly classify pets' emotions in most of the time. Here are 4 example classiifcations.
<p align="center">
  <img src="repo_img/dense_emotion.png"><br>
</p>
We then analyzed the precision, accuracy, and f1-score for each category based on the performance of DenseNet-121 in Experiment 2 on the test dataset. From the table below, we will focus on the prediction on the 'Happy' category more in the next step as its precision and f1-score are relatively low compared to other categories.
<div align="center">
  
| Emotion | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Happy   | 0.6       | 0.75   | 0.67     |
| Angry   | 0.8       | 0.62   | 0.7      |
| Sad     | 0.83      | 0.67   | 0.74     |
| Other   | 0.76      | 0.94   | 0.84     |

</div>

The confusion matrix is generated to further visualize the model prediction on all 4 class (happy, sad, angry, and other) and help us understand the specific misclassification vividly when prediction each categoruy. Additionally, the multi-calss ROC curve is plotted to illustrate the classification ability of DenseNet-121 as its threshold is varied and the AUC provides a single scalar value summarizing the overall performance of the classifier across each emotion category for all thresholds.This ROC-AUC plot further confirmed our model have good performance since the closer the AUC is to 1, the better DenseNet-121 performs.
<p align="center">
  <img src="repo_img/dense_confusion_matrix.png" width="400" height="400"> 
  <img src="repo_img/dense_roc_auc.png" width="400" height="400"> <br>
</p>

#### d. Discussion and Future Work
Our main goal for the next step is to reduce the number of miclassification for each category, as mentioned above. To achieve this, we plan to further experiment modifying the DenseNet-121 architecture by adding some layers at the end, increase the number of epochs to make the model fully learn from the dataset. Additional dataset will be used to make the model more generalizable.

### 3. ResNet50
#### a. Introduction and Advantages
ResNet-50 is a deep convolutional neural network with 50 layers, known for its architecture that features residual connections, which are shortcuts that skip one or more layers. These residual connections help to alleviate the vanishing gradient problem that can occur with traditional deep networks, enabling training of much deeper networks. ResNet-50 utilizes a technique called 'bottleneck layers' to keep the computational load manageable despite its depth.
#### b. Implementation
For this project, we appled the ResNet50 model pre-trained over ImageNet which a custom classification head consisting of global average pooling followed by a dense layer with 3 units and a softmax activation function. The modified model is compiled with the categorical crossentropy loss function and the Adam optimizer. To prevent overfit, EarlyStopping is use which will restore the weights from the best epoch if the validation loss does not improve for 20 consecutive epochs. ReduceLROnPlateau is also used to reduce the learning rate if the validation loss does not improve for 10 epochs. The model is trained over 50 epoches with 10% of trainting data used as validation data.
#### c. Result
From the learning curve and the accuracy curve we can see that the result is not very good underthe current model. The accuracy curve for the validation data fluctuates a lot bettween differnt epoch which means this model is not consistant and robust to the test cases.
<p align="center">
  <img src="repo_img/Learning_curve.png" width="400" height="400"> 
  <img src="repo_img/Accuracy_curve.png" width="400" height="400"> <br>
</p>
The following confusion matrix shows the classification result for each of the happy, sad, and angry class. We can see that the dataset is pretty uneven, and the number of picrtures for angry class is significantly less than the others. A multi-class ROC graph is also included in here, and we can notice that the calssification result is not very good becuase AUCs are pretty far from 1.
<p align="center">
  <img src="repo_img/Confusion_matrix.png" width="400" height="400"> 
  <img src="repo_img/ROC.png" width="400" height="400"> <br>
</p>
The flollowing table shows the precision, recall, and F1 score for each of the class.
<p align="center">
<img src="repo_img/stats.png" width="350" height="150"><br>
</p>

#### d. Discussion and Future Work
The result of ResNet50 over the data set still has a large room for improvement. We might want to adjust to use 100 epoches with better data preprocessing techniques, experiment with more activation functions and optimizers to help to improve the accuracy and F1 score of this model.


## Analysis of 3 Algorithms/Models
### ResNet50:

#### Overview: 
ResNet50 is part of the ResNet (Residual Networks) family, known for its deep network architecture. It consists of 50 layers and is widely recognized for its ability to solve the vanishing gradient problem through the use of skip connections or shortcut connections.

#### Relevance to Project: 
In pet facial expression detection, ResNet50 can effectively handle deep learning challenges, like recognizing subtle features in pet faces. The skip connections help in preserving information even as the network goes deeper, which is crucial for detailed feature extraction in varied pet expressions.

#### Potential Advantages: 
Excellent at handling deeper networks without losing performance, making it suitable for complex tasks like expression recognition. The architecture is robust against overfitting to a certain extent due to its depth.

#### Limitations: 
The sheer size of the network can be a drawback in terms of computational cost and time, especially if the available dataset is not extensive enough to fully utilize its capabilities.

### DenseNet121:

#### Overview:
DenseNet121, part of the DenseNet (Densely Connected Convolutional Networks) family, is characterized by its dense connections, where each layer is connected to every other layer in a feed-forward fashion.

#### Relevance to Project: 
For pet facial expression detection, DenseNet121's feature reuse ability can be highly beneficial. It ensures maximum information flow between layers, which can be crucial for distinguishing between nuanced expressions in pets.

#### Potential Advantages: 
The architecture is highly efficient in terms of computational cost, as it requires fewer parameters than traditional CNNs. It is also less prone to overfitting and can be more accurate due to the improved flow of information.

#### Limitations: 
Dense connections can lead to a large increase in memory footprint, which might be a constraint for resource-limited environments.
### EfficientNetB5:

#### Overview: 
EfficientNetB5 is a part of the EfficientNet family, known for its scalable architecture. It uses a compound scaling method to uniformly scale all dimensions of depth, width, and resolution of the network.
#### Relevance to Project: 
In the context of pet facial expression detection, EfficientNetB5 offers a good balance between accuracy and efficiency. It can handle complex patterns in pet expressions while being resource-efficient.
#### Potential Advantages: 
Provides a balance between performance and computational resources. It's scalable, allowing for adjustments based on the available computing power and dataset size.
#### Limitations: 
The performance heavily depends on the proper scaling of the network. Under-scaling or over-scaling can lead to suboptimal results.

## Comparison: ResNet50 vs DenseNet121 vs EfficientNetB5
### Training & Validation Curves:
Both ResNet50 and DenseNet121 have steady increase over 100 epochs and were able to converge to a high accuracy and low loss on training and validation dataset. However, the EfficientNetB5 was less efficient to converge to a high accuracy. Based on EfficientNetB5's accuracy curves, its training and validation accuracy began to fluctuate between 0.35 and 0.4 after epoch 10. This variation might due to different data preprocessing, data augmentation, different initialization, etc.
### Precision, Recall, and F1-score:
We have precision, recall, and f1-score available for both DenseNet121 and EfficientNetB5. Compared to EfficientNetB5, the average precision difference across all 3 categories (Happy, Sad, Angry) is 0.3033, the average recall difference across all 3 categories is 0.29, and the average f1-score different across all 3 categories is 0.3767. Both DenseNet121 and EfficientNetB5 have very low precision in the category of happy.

Based on the confusion matrix of ResNet50, I calculated the precision, recall, and f1-score.
<div align="center">
  
| Emotion | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Happy   | 0.86      | 0.90   | 0.88     |
| Angry   | 0.86      | 0.89   | 0.87     |
| Sad     | 0.94      | 1.00   | 0.97     |

</div>
Compared to DenseNet121, in ResNet50 the average precision difference across all 3 categories (Happy, Sad, Angry) is -0.1433, the average recall difference across all 3 categories is -0.25, the average f1-score difference across all 3 categories is -0.2033. In contrast with DenseNet121 and EfficientNetB5, ResNet50 has very high precision in the category of happiness. Therefore, based on the precision, recall, and f1-score presented, ResNet50 has the first-place performance, DenseNet121 has the second place performance, and EfficientNetB5 has the third place performance.

### Confusion Matrix:
The confusion matrices for ResNet50, DenseNet121, and EfficientNetB5 provides a visulization of precision, recall, and f1-score mentioned above.

### Next Step:
We can clearly see that every CNN architecture has various performance in each category on different metric. Thus, we plan to first investigate on why EfficientNetB5 has comparatively inferior performance by first extending its training epochs to 100 and then diving deeper into settings like model initializations, etc. 
## Timeline & Responsibility
The Timeline & Responsibility spreadsheet can be accessed through the [link](https://docs.google.com/spreadsheets/d/1KKS6dfy5047rFeydJ3CbfcBmkDKMImo_41aFWszzc4E/edit?usp=sharing).

## Contribution Table

| Student Name | Contribution in Final Report |
|----------|----------|
| Zhujie Xu | Analysis of 3 Algorithms/Models|
| Kuancheng Wang | Comparison: ResNet50 vs DenseNet121 vs EfficientNetB5|
| Pengyu Mo | Video: <br>1.Video recording<br>2.Design and beautification<br>3.Explaining method and evaluation<br>4.Video Editing<br>5.Gantt Chart timeline planning|
| Zexiu An | Video: <br>1.Video recording<br>2.Design and beautification<br>3.Explaining method and evaluation<br>4.Video Editing<br>5.Gantt Chart timeline planning|
| Minkun Lei | Video: <br>1.Video recording<br>2.Design and beautification<br>3.Explaining method and evaluation<br>4.Video Editing<br>5.Gantt Chart timeline planning|

| Student Name | Contribution in Midterm |
|----------|----------|
| Zhujie Xu | Code, visualize, and write the section:<br>1.EfficientNet<br>1(a) Introduction and Advantages<br>1(b) Implementation<br>1(c) Results<br>1(d) Discussion and Future Work |
| Kuancheng Wang | Code, visualize, and write the section:<br>1.PCA<br>2.DenseNet-121<br>2(a) Introduction and Advantages<br>2(b) Implementation<br>2(c) Results<br>2(d) Discussion and Future Work |
| Pengyu Mo | Code, visualize, and write the section:<br>3.ResNet50<br>3(a) Introduction and Advantages<br>3(b) Implementation<br>3(c) Results<br>3(d)  Discussion and Future Work |
| Zexiu An | Completed Dataset Description<br>Completed Timeline/Gantt Chart<br>Proofread and edit the github page to ensure all requirements in midterm checklist are met<br>Edit the format of Github page |
| Minkun Lei | Completed Dataset Description<br>Completed Timeline/Gantt Chart<br>Proofread and edit the github page to ensure all requirements in midterm checklist are met<br>Edit the format of Github page |

| Student Name | Contribution in Proposal |
|----------|----------|
| Zhujie Xu | Write GitHub page: <br>1.motivation<br>2.method & algorithm<br>3.timeline/responsibility<br>4.contribution table<br>5.upload dataset file and attach link |
| Kuancheng Wang | Write GitHub page: <br>1.background<br>2.potential results & discussion<br>3.references<br>4.reformat README |
| Pengyu Mo | Video: <br>1.Video recording<br>2.Design and beautification<br>3.Explaining method and evaluation<br>4.Video Editing<br>5.Gantt Chart timeline planning |
| Zexiu An | Video: <br>1.Video recording<br>2.construct Powerpoint structure and add contents <br>3.Explain motiviation and describe purpose<br>4.README file update and proof-reading<br>5.checking for rubrics regarding github page and video recording |
| Minkun Lei | 1.Video recording<br>2.Check the requirement for this proposal on Ed and Class Homepage, ensuring all components are covered in our github page<br>3.README proof-reading |


# Checkpoints
Checkpoint 1:<br>
10/6: Proposal<br>
Checkpoint 2:<br>
10/29: Finish data prepocess<br>
Checkpoint 3:<br>
11/11: Finish coding model train and test<br>
Checkpoint 4:<br>
11/24: Improve models<br>
Checkpoint 5:<br>
12/5: Write final report<br>


## Reference
<a name="ref1"></a>
[1] Boneh-Shitrit, Tali, et al. "Deep Learning Models for Automated Classification of Dog Emotional States from Facial Expressions." arXiv preprint arXiv:2206.05619 (2022).  
<a name="ref2"></a>
[2] Sinnott, Richard O., et al. "Run or pat: using deep learning to classify the species type and emotion of pets." 2021 IEEE Asia-Pacific Conference on Computer Science and Data Engineering (CSDE). IEEE, 2021.    
<a name="ref3"></a>
[3] Mao, Yan, and Yaqian Liu. "Pet dog facial expression recognition based on convolutional neural network and improved whale optimization algorithm." Scientific Reports 13.1 (2023): 3314.  
<a name="ref4"></a>
[4] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.  
<a name="ref5"></a>
[5] Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
