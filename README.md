# Facies Classification

## Introduction
Facies Classification Using a new Deep Learning Method

## Requirements
Python libraries requirments:

```bash
PyTorch >= 2.0.0
transformers >= 4.41.2
scikit-learn >= 1.5.1
```

## Data Processing

Data can be downloaded from [repo](https://github.com/mardani72/Facies-Classification-Machine-Learning/tree/master) 

We save data in `./data`

Processing data is developed in `./data/process_data.py`

```bash
python ./data/process_data.py
```

## Implementation

- **Proposed Architecture**: `model.py` .
- **ML models**: `ML_models.py`.
- **Metrics**: `utils.py`.

## Training model
```bash
python running.py
```