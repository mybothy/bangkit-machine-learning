# Hyfit ML Documentation
## Overview
This repository contains the documentation and resources for the Hyfit ML project, which involves training models using transfer learning techniques on a custom dataset of food images.

## Dataset
The dataset is located in the data folder. It consists of images of 3 different types of food sourced from [Kaggle's "101 Food"](https://www.kaggle.com/datasets/dansbecker/food-101/data) dataset.

## Data Testing
We collected several [sample images](https://github.com/mybothy/bangkit-machine-learning/tree/main/tes) from the internet  to test the model's performance.

## Training the First Model
### Model 1:
- **Description**: Utilizes transfer learning with EfficientNet-B0 but with frozen layers.
- **Files**:
  - [`imageclassification2.0.ipynb`](imageclassification2.0.ipynb)
  - [`convert`](convert)
- **Result**: 
- **How to Use**: Open `imageclassification2.0.ipynb` in Jupyter Notebook, follow the instructions to set up the environment and run the model training.

## Training the Second Model
### Model 2:
- **Description**: Employs the same transfer learning approach but fine-tunes more layers within the CNN infrastructure.
- **File**:
  - [`imageclassification2.0.ipynb`](imageclassification2.0.ipynb)
  - [`convert2`](convert2)
- **How to Use**: Open `imageclassification2.0.ipynb` in Jupyter Notebook, follow the instructions to set up the environment and run the model training.

## Training the Third Model
Model 3:
Uses transfer learning with additional fine-tuning and unfreezing of some last layers.
The files FINALBGSBGT.ipynb and ipynb3.0 contain the implementation for this model.

## FINALBGSBGT
This folder includes the final model, which has been optimized with several enhancements.
It also contains documentation and metadata for the TensorFlow Lite model. The trytflitemodel.py script is used to create the metadata, while testmetadata.py is used for testing the TFLite model with the added metadata.

## How To Run?
Clone this repository to your local machine.
Navigate to the desired model's notebook file (convert.ipynb, convert2.ipynb, or FINALBGSBGT.ipynb).
Follow the instructions within each notebook to set up and run the model training process.
For testing the models, use the respective scripts (trytflitemodel.py and testmetadata.py) located in the FINALBGSBGT folder.
