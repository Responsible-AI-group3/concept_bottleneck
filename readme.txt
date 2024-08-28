# Bird Dataset Concept Bottleneck Project

This project implements a custom dataset class for the CUB-200-2011 dataset, focusing on concept bottleneck models.

## Installation

### Prerequisites

- Python 3.9.18 or higher
- pip (Python package installer)





#### On Linux/macOS and DTU hpc:

 Create a virtual environment:
   ```
   python -m venv env
   ```

Activate the virtual environment:
   ```
   source env/bin/activate
   ```

#### On Windows:


Create a virtual environment:
   ```
   python -m venv env
   ```
 Activate the virtual environment:
   ```
   .\env\Scripts\activate
   ```

### Installing Dependencies

Once your virtual environment is activated, install the required packages:

```
pip install -r requirements.txt
```

## Usage

After installation, you can import and use the BirdDataset class in your Python scripts:

```python
from bird_dataset import BirdDataset

# Create a dataset instance
dataset = BirdDataset(data_dir='path/to/CUB_200_2011', 
                      train=True, 
                      majority_voting=True, 
                      concept_threshold=0.05)