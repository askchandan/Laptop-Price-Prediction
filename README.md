# Laptop Price Prediction

## Overview

This project aims to predict the price of laptops based on various features such as brand, processor type, RAM, storage, and screen size. The project includes data preprocessing, exploratory data analysis, model building using machine learning techniques, and deployment of the model as a web application using Flask.

## Data Description

The dataset used for this project is `laptop_data.csv`, which contains the following features:

-   **Unnamed: 0**: Index
-   **Company**: Laptop manufacturer
-   **TypeName**: Type of laptop (e.g., Ultrabook, Gaming)
-   **Inches**: Screen size in inches
-   **ScreenResolution**: Screen resolution details
-   **Cpu**: Processor details
-   **Ram**: RAM size in GB
-   **Memory**: Storage details
-   **Gpu**: Graphics card details
-   **OpSys**: Operating system
-   **Weight**: Weight of the laptop in kg
-   **Price**: Price of the laptop (target variable)

## Project Structure

```
Laptop-Price-Prediction/
├── .gitignore
├── Laptop-Price-Predictor.ipynb  # Jupyter Notebook with EDA and model building
├── app.py                        # Flask application for deployment
├── df.pkl                        # Pickled DataFrame for faster loading
├── laptop_data.csv               # Dataset
├── pipe.pkl                      # Pickled machine learning pipeline
├── README.md                     # This file
├── requirements.txt              # Project dependencies
└── LICENSE                       # License information
```

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/askchandan/Laptop-Price-Prediction.git
    cd Laptop-Price-Prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preprocessing and Model Building

-   Open and run the `Laptop-Price-Predictor.ipynb` Jupyter Notebook to perform data cleaning, exploratory data analysis, feature engineering, and model training.
-   The notebook saves the preprocessed DataFrame as `df.pkl` and the trained pipeline as `pipe.pkl`.

### 2. Running the Flask Application

1.  **Set environment variables (optional):**

    -   You can set environment variables for the port and debug mode. If not set, the app defaults to port 5000 and debug mode off.

    ```bash
    export PORT=8000  # Example
    export DEBUG=1    # Example (1 for True, 0 for False)
    ```

2.  **Run the Flask app:**

    ```bash
    python app.py
    ```

3.  **Access the application:**

    -   Open your web browser and go to `http://127.0.0.1:5000` (or the port you specified).

### 3. Making Predictions

-   Use the web interface to input the laptop's specifications.
-   Click the "Predict Price" button to see the predicted price.

## Model Details

-   The machine learning pipeline (`pipe.pkl`) includes data preprocessing steps and a regression model for price prediction.
-   The model is trained on the `laptop_data.csv` dataset.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
