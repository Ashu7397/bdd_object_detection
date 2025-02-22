# bdd_object_detection
Object Detection on BDD100K Dataset

# BDD Object Detection Docker Setup

This repository contains the code for BDD Object Detection Dataset analysis. The code is containerized using Docker to ensure it can run on any machine.

## Prerequisites

- Docker installed on your machine. You can download it from [here](https://www.docker.com/products/docker-desktop).
- **IMPORTANT** Make sure the bdd data is stored in the folder named assignment_data_bdd in the same directory.

## Building the Docker Image

1. Clone the repository:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Build the Docker image:

    ```sh
    docker build -t bdd-object-detection .
    ```

## Running the Docker Container

1. Run the Docker container:

    ```sh
    docker run -p 8888:8888 -v $(pwd):/app bdd-object-detection
    ```

## Data Flow

The data flow within this project is structured as follows:

1.  **Dataset**: The BDD100K dataset is expected to be located in the `assignment_data_bdd` folder, structured with images and labels.
2.  **Data Loading (`data.py`)**:
    *   The `BDDObjectDetectionDataset` class handles loading and preprocessing the BDD100K dataset.
    *   It supports splitting the data into training and validation sets, specified during dataset initialization.
    *   Annotations are loaded from JSON files and cached as parquet files for faster loading in subsequent runs.
    *   The dataset class provides methods to access images and their corresponding bounding box annotations.
3.  **Data Transformation (`data.py`)**:
    *   The `__getitem__` method retrieves an image and its target (bounding boxes and labels).
    *   The `custom_collate_fn` function is used to collate batches of data, converting PIL Images to tensors.

## Training Flow

The training process is managed by `train_model.py`. Here's a breakdown of the key steps:

1.  **Data Loading (`train_model.py`)**:
    *   The `train_function` initializes the training and validation datasets using `BDDObjectDetectionDataset`.
    *   `DataLoader` is used to create iterable data loaders for training and validation sets, using `custom_collate_fn` to handle batching.

2.  **Model Definition (`model.py`)**:
    *   The `ObjectDetectionModel` class defines the object detection model as a PyTorch Lightning module.
    *   It uses a pre-trained Faster R-CNN model with a ResNet-50 backbone, obtained from `torchvision.models`.
    *   The `get_pretrained_model` function configures the model, replacing the classifier with a new one suitable for the BDD100K dataset (10 object classes + background).
    *   The backbone's pre-trained weights are frozen during initial training to stabilize training and leverage pre-trained features.

3.  **Training Loop (`train_model.py` and `model.py`)**:
    *   The `train_function` sets up the training and validation data loaders, the model, and the PyTorch Lightning trainer.
    *   **Loss Function (`losses.py`)**: The `ObjectDetectionLoss` class calculates the combined loss (classification and regression) for the object detection task. It includes Focal Loss for classification and Smooth L1 Loss for bounding box regression.
    *   **Optimizer (`model.py`)**: The Adam optimizer is used with a learning rate of 1e-3.
    *   **Callbacks (`train_model.py`)**:
        *   `ModelCheckpoint`: Saves the best model based on validation loss. Checkpoints are saved in the `checkpoints/` directory. The filename includes the epoch number and validation loss.
        *   `EarlyStopping`: Stops training when the validation loss stops improving, with a patience of 10 epochs.
    *   **Logging (`train_model.py`)**: TensorBoard is used for logging metrics during training. Logs are stored in the `lightning_logs` directory.
    *   The `training_step` and `validation_step` methods in `ObjectDetectionModel` define the training and validation logic, respectively.

4.  **Running Training (`train_model.py`)**: To start the training, execute the `train_model.py` script:

    ```sh
    python bdd_object_detection/train_model.py
    ```

## Result Analysis Flow

The `result_analysis.ipynb` notebook provides a comprehensive analysis of the model's performance. Here's a breakdown of the key steps:

1.  **Data Loading (`result_analysis.ipynb`)**:
    *   Prediction and ground truth data are loaded from parquet files (`bdd100k_val_cache_predictions.parquet` and `bdd100k_val_cache.parquet`, respectively).
    *   These files should contain the bounding box predictions and ground truth annotations for the validation set.

2.  **Metric Calculation (`losses.py` and `result_analysis.ipynb`)**:
    *   The `calculate_metrics` function in `losses.py` calculates object detection metrics such as Precision, Recall, and F1-score based on Intersection over Union (IoU) between predicted and ground truth bounding boxes.
    *   The notebook iterates through different score thresholds and IoU thresholds to evaluate the model's performance under various conditions.
    *   Metrics are calculated for each category and for all categories combined.

3.  **Visualization (`result_analysis.ipynb`)**:
    *   The notebook generates various plots to visualize the model's performance:
        *   **Precision-Recall curves**: Plots of precision vs. recall for different IoU thresholds.
        *   **F1-score curves**: Plots of F1-score vs. score threshold for different IoU thresholds.
        *   **Bar charts**: Bar charts comparing precision, recall, and F1-score for different categories at a fixed IoU threshold.
        *   **AP (Average Precision) analysis**: Analysis of AP, AP50, AP75, and AR (Average Recall) metrics, including plots for individual classes and overall performance.
    *   The plots help to identify the optimal score threshold for each category and to understand the model's strengths and weaknesses.

4.  **Max F1 Score Analysis (`result_analysis.ipynb`)**:
    *   The notebook identifies the score threshold at which the F1-score is maximized for each category.
    *   This allows for setting a working point for the model where it performs the best, balancing precision and recall.

5.  **Average Precision Analysis (`result_analysis.ipynb`)**:
    *   The notebook analyzes the Average Precision (AP) for each class using pre-computed results from a RetinaNet model.
    *   It generates bar charts to visualize the AP for different classes and IoU thresholds.

6.  **Running Analysis (`result_analysis.ipynb`)**:
    *   To run the analysis, execute the `result_analysis.ipynb` notebook in a Jupyter environment.

## Notes

-   The `Dockerfile` sets up the environment and installs all necessary dependencies specified in `requirements.txt`.
-   This `README.md` provides instructions on how to build and run the Docker container, train the model, and analyze the results