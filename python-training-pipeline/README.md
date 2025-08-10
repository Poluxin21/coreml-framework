# Python Training Pipeline

This project implements a training pipeline for a machine learning model using Python. It includes data generation, model training, and ONNX export functionalities, designed to work alongside a C++ core framework.

## Project Structure

```
python-training-pipeline
├── data
│   └── generate_data.py       # Script for generating synthetic training data
├── models
│   └── model.py               # Defines the model architecture
├── scripts
│   ├── train.py               # Training pipeline script
│   └── export_onnx.py         # Exports the trained model to ONNX format
├── core_cpp
│   └── README.md               # Documentation for the C++ core framework
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and usage instructions
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd python-training-pipeline
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Generate Data**:
   Run the data generation script to create synthetic datasets:
   ```
   python data/generate_data.py
   ```

2. **Train the Model**:
   Execute the training script to train the model:
   ```
   python scripts/train.py
   ```

3. **Export to ONNX**:
   After training, export the model to ONNX format:
   ```
   python scripts/export_onnx.py
   ```

## Additional Information

Refer to `core_cpp/README.md` for details on the C++ core framework and how it integrates with this Python training pipeline.