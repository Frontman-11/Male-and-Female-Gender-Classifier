# Male and Female Gender Classifier

## Project Description

The aim of this project is to accurately classify images into male and female genders using a Convolutional Neural Network (CNN). The model can be particularly useful for dataset labeling, such as in the VGGFace2 dataset with over 3 million instances. It can also serve as a base model for face recognition or n-shot learning problems.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Repository Structure](#repository-structure)
- [License](#license)

## Installation

This project was developed on Kaggle but can be set up locally as well. To set it up locally:

1. Clone the repository.
2. Create a virtual environment and activate it:
   - On Windows:  
     ```bash  
     python -m venv env  
     .\env\Scripts\activate  
     ```
   - On macOS/Linux:  
     ```bash  
     python3 -m venv env  
     source env/bin/activate  
     ```
3. Install the required dependencies using:  
   ```bash  
   pip install -r requirements.txt  
   ```

## Usage

1. **Dataset Preparation:**  
   - Download the [raw dataset](https://www.kaggle.com/datasets/yasserhessein/gender-dataset).
   - Run the `tfrecord_writing.ipynb` notebook to convert the raw dataset into TFRecord format, which will be saved to the `data/processed` directory.  
   - Make sure to edit the file paths in the `config.py` module to accurately reflect your directories. Alternatively, download the processed dataset from [this link](https://www.kaggle.com/work/collections/14474385) on Kaggle if you don't want to convert the dataset yourself.

2. **Training the Model:**  
   - The training process consists of two stages: initial training and final training.

   **Stage 1: Initial Training**  
   - For the training set (obtained from `preprocess.gender_dataset`):  
     - Set `repeat` to 2.  
     - Set `normalize` to `False`.  
     - Optionally adjust `batch_size` based on your available resources.
   - Configure the callback parameters:  
     - Set `ReduceLROnPlateau` patience to 3.  
     - Set `EarlyStopping` patience to 5.
   - In `train.train_model`:  
     - Set `epochs` to 25 (early stopping will terminate early if needed).  
     - Set `with_aug_layer` to `True` to apply random augmentation (note that normalization will also be applied, which is why `normalize` must be set to `False` for the training set).  
     - Set `insert_dropout` to `None`.  
     - Optionally change `save_model_to` and `show_model_summary` from their default values.  
     - This model should achieve at least 98.18% accuracy on the validation set.

   **Stage 2: Final Training**  
   - For the training set (obtained from `preprocess.gender_dataset`):  
     - Set `repeat` to `None` or `False`.  
     - Set `normalize` to `True`.  
     - Optionally adjust `batch_size` based on your available resources.
   - Configure the callback parameters:  
     - Set `ReduceLROnPlateau` patience to 2.  
     - Set `EarlyStopping` patience to 3.
   - In `train.train_model`:  
     - Set `epochs` to 10 or more (early stopping will terminate early if needed).  
     - Set `with_aug_layer` to `False`.  
     - Set `insert_dropout` to `tf.keras.layers.Dropout(rate=0.3, name='dropout_layer_2')`.  
     - Optionally change `save_model_to` and `show_model_summary` from their default values.
   - Ensure you rerun the script for the second stage to make sure the last trained weights are correctly updated before beginning training. At this stage the model should be able to get to 98.46 accuracy on the validation set which is the final model used in this project.

3. **Evaluation:**  
   - Run `evaluation_notebook.ipynb` to evaluate the model. Due to the size of the generated picture grid, it's recommended to save the picture to the output directory for analysis rather than displaying it directly in the notebook. Checkout [wrong prediction grid](./output/wrong_preds.md) for the wrong prediction grid of the final model on all three datasets.

## Model Details

This model is a custom-built Convolutional Neural Network (CNN) designed from scratch, not a fine-tuned version of any existing model. It follows a sequential architecture with stacked layers, including convolutional layers, batch normalization, activation functions, and max pooling. The model concludes with two dense layers for classification. For more details and diagram on the architecture, [click here](./output/model_1.keras.png).

## Repository Structure

- `data/`: Contains data-related files.
- `src/`: Contains Python scripts for preprocessing, model training, evaluation, etc.
- `models/`: Contains saved models, weights, and architecture files.
- `.gitattributes`: Configuration for Git.
- `.gitignore`: Specifies files to be ignored by Git.
- `LICENSE`: Licensing information.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of dependencies.

## License  
This project is licensed under the MIT License.
