# Hyfit ML Documentation
## Overview
This repository contains the documentation and resources for the Hyfit ML project, which involves training models using transfer learning techniques on a custom dataset of food images.

## Built With
* [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)][TensorFlow-url]
* [![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54)][Python-url]
* [![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=flat&logo=numpy&logoColor=white)][NumPy-url]
* [![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)][TFLite-url]

[TensorFlow-url]: https://www.tensorflow.org/
[Python-url]: https://www.python.org/
[NumPy-url]: https://numpy.org/
[TFLite-url]: https://www.tensorflow.org/lite



## Dataset
The dataset is located in the data folder. It consists of images of 3 different types of food sourced from [Kaggle's "101 Food"](https://www.kaggle.com/datasets/dansbecker/food-101/data) dataset.

## Data Testing
We collected several [sample images](https://github.com/mybothy/bangkit-machine-learning/tree/main/tes) from the internet  to test the model's performance.

## Transfer Learning
We use [EfficienNetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0) for our model transfer learning with some added new layer and also fine tuning it

## Training the First Model
### Model 1:
- **Description**: Utilizes transfer learning with EfficientNet-B0 but with frozen layers.
- **Files**:
  - [`imageclassification2_0.ipynb`](imageclassification2_0.ipynb)
  - [`convert`](convert)
- **Result**: 
- **How to Use**: Open `imageclassification2.0.ipynb` in Colab, follow the instructions to set up the environment and run the model training.

## Training the Second Model
### Model 2:
- **Description**: Employs the same transfer learning approach but fine-tunes more layers within the CNN infrastructure.
- **File**:
  - [`imageclassification2_0.ipynb`](imageclassification2_0.ipynb)
  - [`convert2`](convert2)
- **How to Use**: Open `imageclassification2_0.ipynb` in Colab, follow the instructions to set up the environment and run the model training.

## Training the Third Model
### FINALBGSBGT (Model 3)
- **Description**: This folder includes the final model, which has been optimized with several enhancements. It also contains documentation and metadata for the TensorFlow Lite model.
- **Files**:
  - [`trytflitemodel.ipynb`](trytflitemodel.ipynb) (creates the metadata for the TFLite model)
  - [`testmetadata.ipynb`](testmetadata.ipynb) (tests the TFLite model with the added metadata)
  - [`makemetadata2.0.ipynb`](makemetadata2.0.ipynb) (to create the metadata without normalization)
- **How to Run**: Navigate to the [`FINALBGSBGT`](FINALBGSBGT) folder, and run the scripts as follows:
  - [`trytflitemodel.ipynb`](trytflitemodel.ipynb) to create the metadata
  - [`testmetadata.ipynb`](testmetadata.ipynb) to test the TFLite model with metadata.
  - [`makemetadata2.0.ipynb`](makemetadata2.0.ipynb) to create the metadata without normalization

## FINALBGSBGT
- This folder [`FINALBGSBGT`](FINALBGSBGT) includes the final model, which has been optimized with several enhancements.
- It also contains documentation and metadata for the TensorFlow Lite model. The trytflitemodel.ipynb script is used to create the metadata, while testmetadata.ipynb is used for testing the TFLite model with the added metadata.
- In this folder there is our last final model ['my_model_lite_with_metadata.tflite'](FINALBGSBGT/my_model_lite_with_metadata.tflite) that contain metadata too with labels.txt as the class name

## Performance
![Accuracy](https://github.com/mybothy/bangkit-machine-learning/blob/main/FINALBGSBGT/download%20(2).png "App Screenshot")
![Test and Pred](https://github.com/mybothy/bangkit-machine-learning/blob/main/FINALBGSBGT/download%20(5).png "App Screenshot")


<!-- ## How To Run?
Clone this repository to your local machine.
Navigate to the desired model's notebook file (convert.ipynb, convert2.ipynb, or FINALBGSBGT.ipynb).
Follow the instructions within each notebook to set up and run the model training process.
For testing the models, use the respective scripts (trytflitemodel.py and testmetadata.py) located in the FINALBGSBGT folder.-->
