# Smurf Detection in Leage of Legends

This model is designed to detect player anomalies or smurfs in the game League of Legends.

## 📁 Project Structure
- `Full.py`: The main execution script.
- `cut_model.keras`, `pool_model.keras`, `raw_model.keras`: Pre-trained models used to save training time found in the models folder.
- `Data_Editing.py` / `import_scraper.py`: Scripts used for data collection and preparation (Reference only).

## 📊 Data Access
The required NumPy datasets are hosted on Google Drive due to their size.
**[Link to Google Drive Data]** (Insert your link here)

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the environment set up with all necessary packages. Use the provided environment file: requirements.txt
Go into the correct directory in terminal then use the command:
pip install -r requirements.txt
To install the correct packages

### 2. File Setup & Preparation
To run the code, you must organize the files into a single directory:

1.  **Download Data:** Download the **two .npy files** from the Google Drive link provided above.
2.  **Models:** Ensure the three `.keras` files (`cut_model.keras`, `pool_model.keras`, and `raw_model.keras`) are in the same folder.
3.  **Scripts:** Ensure `Full.py` is in that same folder.

### 3. Running the Code

**Performance Note:** This code can be computationally intensive and may require significant processing time depending on your hardware specifications. Please ensure your system has adequate resources available before execution.


Once your folder contains the models, scripts, and the two `.npy` data files, execute the application:

```bash
python Full.py
