# Pneumonia Detection using FastAI ResNet50 Model
This project uses a pre-trained ResNet50 model from the FastAI library to detect pneumonia in chest X-rays. The dataset which is available on kaggle is used for training the model which classifies the chest xray as NORMAL, VIRAL PNEUMONIAL or BACTERIAL PNEUMONIAL

First, train the model to generate the export.pkl file. Once the training process is complete, you can use this generated export.pkl file in your Flask application.

Before uploading image
<image  width="600px" src="Screenshots/initial.jpg">

Uploading the image 
<image  width="600px" src="Screenshots/middle.jpg">

After uploading the image 
<image  width="600px" src="Screenshots/final.jpg">

### Getting Started
To use this project, follow these steps:
1) Clone the repository: git clone https://github.com/Divyam6969/Pneumonia-Detection-using-FastAI<br>
2) Install the necessary dependencies: pip install fastbook<br>
3) Download the Chest X-ray dataset from Kaggle and extract it to the appropriate directory.<br>
Open the pneumonia-detection.ipynb notebook and follow the instructions to train and evaluate the model.<br>

Note: You will need to have FastAI installed to run this project.

### Features
Uses a pre-trained ResNet50 model for Pneumonia detection.
Demonstrates how to fine-tune a pre-trained model using FastAI.
Achieves high accuracy in detecting pneumonia in chest X-rays.
Includes visualizations of the training process, confusion matrix, and sample predictions.

### Contribution
Contributions and feedback are welcome. Please open an issue or a pull request if you have any suggestions or improvements.

### Resources
Link to another project I made which classifies chest-xray as pneumonic or normal: https://www.kaggle.com/code/divyam6969/chest-xray-classifier

Link to my previous project: https://github.com/Divyam6969/Pneumonia-Detection-AI (this model is overfitting :/)

Chest X-ray dataset on Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

FastAI library: https://www.fast.ai/

Kaggle API: https://github.com/Kaggle/kaggle-api

### License
This project is licensed under the MIT License. See the <a href="https://github.com/Divyam6969/Pneumonia-Detection-using-FastAI-/blob/main/LICENSE">LICENSE</a> file for details.

