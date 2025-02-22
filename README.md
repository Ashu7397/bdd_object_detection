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

2. After running the above command, you will see an output similar to this:

    ```sh
    [I 13:45:23.456 NotebookApp] Serving notebooks from local directory: /app
    [I 13:45:23.456 NotebookApp] The Jupyter Notebook is running at:
    [I 13:45:23.456 NotebookApp] http://0.0.0.0:8888/?token=<token>
    ```

3. Open the provided URL in your browser to access the Jupyter Notebook. The URL will look something like this:

    ```
    http://localhost:8888/?token=<token>
    ```

## Running the Code

1. Open the `analysis_notebook.ipynb` notebook in Jupyter.
2. Run the cells in the notebook to perform the analysis.

## Notes

- The `Dockerfile` sets up the environment and installs all necessary dependencies.
- The `requirements.txt` file lists all the Python packages required for the project.
- The `README.md` provides instructions on how to build and run the Docker container.
