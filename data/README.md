# Elliptic dataset loader (BF-4)

Raw CSV files are NOT stored in this repo (the features file is ~670 MB).
They live in the shared Google Drive folder `Explainable-AML-GAT-ResNet/data`
(original source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).

## One-time setup
1. Ask the owner to share the Drive folder with your Google account.
2. In Google Drive: right-click the shared folder -> Organize -> Add shortcut -> My Drive.
   Keep the folder name `Explainable-AML-GAT-ResNet`.

## Run in Colab
```python
from google.colab import drive
drive.mount('/content/drive')

# Clone the repo (use your own GitHub token if the repo is private)
!git clone https://github.com/thuyduong955/Explainable-AML-GAT-ResNet.git /content/repo
%cd /content/repo
!python data/load_elliptic.py
```
Expected last line: `All checks passed: 203,769 nodes, 166 columns besides txId.`

## Use in your own code
```python
import sys
sys.path.append("/content/repo/data")
from load_elliptic import load_elliptic
features, edges, classes = load_elliptic()
```

## Data layout
| DataFrame | Shape | Notes |
|---|---|---|
| `features` | (203769, 167) | No header. Column 0 = txId, column 1 = time step, columns 2-166 = 165 features. "166 features" counts everything except txId. |
| `edges` | (234355, 2) | Columns `txId1`, `txId2` |
| `classes` | (203769, 2) | `class`: `1` = illicit, `2` = licit, `unknown` |

To use another data location, set the `ELLIPTIC_DIR` environment variable.
