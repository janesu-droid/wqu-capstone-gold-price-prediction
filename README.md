# Gold Price Prediction — WQU Capstone

Ensemble deep learning system for gold price forecasting, combining CNN-BiLSTM, multi-scale feature extraction, Hidden Markov Model regime detection, and Transformer-based models. Developed as the final capstone project for the WorldQuant University MScFE program.

## Project structure

```
M7_WQU_Final.ipynb          # Main notebook — full pipeline from data to results
requirements.txt            # Python dependencies
cot_gold.csv                # CFTC Commitment of Traders — gold positioning data
WGC_Central_Bank_Reserves.csv  # World Gold Council — central bank gold reserves
```

## Requirements

- Python 3.11.7
- pip

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/emi-lex/wqu-capstone-gold-price-prediction.git
cd wqu-capstone-gold-price-prediction
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

**macOS (Apple Silicon or Intel):**
```bash
pip install -r requirements.txt
```

**Linux / Windows:**

The `requirements.txt` lists `tensorflow-macos` and `tensorflow-metal` for macOS GPU support. On Linux or Windows, replace those two lines with `tensorflow==2.16.2` before installing:

```bash
# Edit requirements.txt: replace tensorflow-macos and tensorflow-metal with:
#   tensorflow==2.16.2
pip install -r requirements.txt
```

### 4. Required data files

Two data files must be present in the project root before running the notebook:

| File | Source |
|------|--------|
| `cot_gold.csv` | CFTC Commitment of Traders report (gold futures) |
| `WGC_Central_Bank_Reserves.csv` | World Gold Council — central bank statistics |

These files are included in the repository.

### 5. Open and run the notebook

Open `M7_WQU_Final.ipynb` in your IDE (VS Code, PyCharm/DataSpell, JupyterLab, etc.), select the `.venv` kernel, and run all cells.

## Configuration

Key parameters are defined at the top of the config cell in the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FAST_RUN` | `False` | Set to `True` for a quick smoke test (2 epochs, 1 CV split) |
| `EPOCHS` | `40` | Training epochs per model |
| `LOOKBACK` | `60` | Sequence look-back window (days) |
| `N_SPLITS_WF` | `3` | Walk-forward cross-validation splits |
| `STRICT_REQUIRED_INPUTS` | `True` | Fail if COT / WGC data files are missing |

