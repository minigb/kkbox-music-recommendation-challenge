# LightGBM Method

## Environment Setup

1. **Install Dependencies**  
   Run the following command in a virtual environment to install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Python Version**  
   Ensure you are using **Python 3.11.4**.

---

## Dataset Setup

1. Download the dataset from the [KKBox Music Recommendation Challenge](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data).  
2. Place the downloaded dataset in the `dataset` directory, ensuring that the `kkbox` directory is placed within it.

---

## Running the Main Script

1. Run the main script using:  
   ```bash
   python scripts/main_script.py
   ```

2. **Try Different Settings**  
   - Modify configurations in the `config.yaml` file to adjust feature engineering and model settings.  
   - To run all possible feature engineering combinations, use the `--run_all` option:  
     ```bash
     python scripts/main_script.py --run_all
     ```

---

## Result Analysis

1. Analyze the results by running the following Jupyter notebook:  
   ```bash
   output_lgbm/result_analysis.ipynb
   ```
2. This will generate analysis files in both JSON and CSV formats.

---

## Kaggle Submission

1. To submit your results to Kaggle, use the following command:  
   ```bash
   python scripts/submit_best_model.py
   ```  
   This script automatically performs inference using the best model specified in `output_lgbm/best_model.json` and submits the results to Kaggle.

---

# Random Forest

To use the Random Forest model, run the following Jupyter notebook:  
```bash
RS_Project_randomforest.ipynb
```

---

# Collaborative Filtering

1. Check out the following files for collaborative filtering:  
   - `collaborative_filtering_NN_3.py`  
   - `latent_factor_model_2.py`  

2. When running these files, ensure the KKBox dataset-related CSV files are placed directly under the `dataset` directory.