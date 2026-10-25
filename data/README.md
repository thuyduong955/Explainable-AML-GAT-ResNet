# Elliptic dataset loader (BF-4)

Raw CSV files are NOT stored in this repo (the features file is ~670 MB).
They live in the shared Google Drive folder `Explainable-AML-GAT-ResNet/data`.

## One-time setup
1. Ask the owner to share the Drive folder with your Google account.
2. In Google Drive: right-click the shared folder -> Organize -> Add shortcut -> My Drive.
   Keep the folder name `Explainable-AML-GAT-ResNet`.

## Run in Colab
```python
from google.colab import drive
drive.mount('/content/drive')
!python data/load_elliptic.py
```

## Use in your own code
```python
import sys
sys.path.append("/content/repo/data")
from load_elliptic import load_elliptic
features, edges, classes = load_elliptic()
```
Features: column 0 = txId, column 1 = time step, columns 2-166 = features.
To use another location, set the ELLIPTIC_DIR environment variable.
