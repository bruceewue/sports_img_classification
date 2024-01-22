# MLzoomcamp Capstone 2 - 100 Sports Image Classification

## Problem Description

The ["100 Sports Image Classification"](https://www.kaggle.com/datasets/gpiosenka/sports-classification/data) dataset presents a challenge for developing a machine learning model capable of accurately classifying images across 100 different sports categories. The dataset is meticulously curated to provide a high-quality resource for model training and validation. It consists of 13,493 training, 500 testing, and 500 validation images, all standardized to a size of 224x224 pixels in JPG format. 

## Dataset Overview

- **Total Images**: 14,493 (13493 train, 500 test, 500 validate)
- **Format**: JPG
- **Image Dimensions**: 224x224x3

## 1. Exploratory Data Analysis (EDA)

### Overview
The EDA for the Sports Classification project is detailed in the Jupyter Notebook titled "sports-classification.ipynb". This notebook includes an in-depth analysis of the sports image dataset used in this project. The EDA section covers the following key aspects:

- Data distribution and balance across different sports categories.
- Visual exploration of sample images from each category.
- Identification of potential data augmentation strategies.

Refer to the "sports-classification.ipynb" for detailed EDA steps and findings.

## 2. Model Training

### Overview
Model training is documented in "sports-classification.ipynb". The process involves using Optuna for hyperparameter optimization. Key highlights include:

- Building a convolutional neural network model using TensorFlow and Keras.
- Employing EfficientNetB0 as the base model with custom top layers.
- Using Optuna to find the optimal hyperparameters such as learning rate, dropout rate, and additional dense layers.
- Training the model with augmented image data to enhance generalization.

Refer to the "sports-classification.ipynb" for detailed training procedures and results.

## 3. Reproducibility & Dependency and Environment Management

### Cloning the Project
To clone the project, use the following Git command:
```
git clone <repository-url>
```

### Environment Setup using Poetry
The project uses Poetry for dependency and environment management. Follow these steps to set up:

1. Install Poetry:
   ```
   curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
   ```
2. Inside the cloned project directory, install dependencies:
   ```
   poetry install
   ```

### Running the Jupyter Notebook
After setting up the environment, launch Jupyter Notebook:
```
poetry run jupyter notebook
```
Navigate to "sports-classification.ipynb" and run the cells to train the model. The trained model will be automatically saved locally.

## 4. Containerization

### Building a Docker Image
The project includes a Dockerfile for containerization. To build the Docker image, run:
```
docker build -t sports-classification .
```

### Publishing to Amazon ECR
Push the built image to Amazon Elastic Container Registry (ECR) using:
```
docker tag sports-classification:latest <ecr-repository-url>
docker push <ecr-repository-url>
```

## 5. Cloud Deployment

### Deploying with AWS Lambda
The Docker image can be used to create a service with AWS Lambda. Follow AWS documentation for deploying a Docker container on Lambda.

The following image showcases the deployment screen of the Sports Classification model on AWS Lambda.

![AWS-Lambda](/images/lambda.png)



### Testing the Endpoint
The deployed model can be tested with the following curl command:
```
curl -X POST -H "Content-Type: application/json" \
     -d '{"url": "https://i.ibb.co/tYRBfFR/2.jpg"}' \
     "https://qmvv6ahdp9.execute-api.eu-west-2.amazonaws.com/test/predict/"
```

Alternatively, use the `test.py` script included in the project for testing:
```
python test.py
```

The `test.py` script will send a request to the Lambda endpoint with a test image URL and display the prediction result.

