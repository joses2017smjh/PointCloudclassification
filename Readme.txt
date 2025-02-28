The interactive menu provides the following options:
1. View all classes (one sample each)
2. View specific class
3. View multiple samples from specific class
4. Switch dataset split (train/test)
5. Exit

## Dataset Details

- **ModelNet10**: A subset containing 10 categories
- **ModelNet40**: Full dataset with 40 categories
- Each category contains:
  - Training samples
  - Testing samples
  - OFF format 3D models
  - Normalized point

## Installation 
Create a virtual environment named "env"
python -m venv env
Activate the virtual environment:
On Windows:
env\Scripts\activate
On macOS/Linux:
source env/bin/activate
Install required packages
pip install torch torchvision matplotlib numpy open3d

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Open3D

## Note

The datasets (ModelNet10 and ModelNet40) are not included in this repository due to their size. Please download them separately from the [ModelNet website](https://modelnet.cs.princeton.edu/).

## License

This project is for educational and research purposes. The ModelNet dataset has its own license terms - please refer to the [ModelNet website](https://modelnet.cs.princeton.edu/) for more information.
