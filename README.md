# Environment Setup

1. **Install Dependencies**  
   Run the following command in a virtual environment to install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Python Version**  
   Ensure you are using **Python 3.11.4**.

---

# Dataset Setup

1. Download the dataset from the [KKBox Music Recommendation Challenge](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data).  
2. Place the downloaded dataset in the `dataset` directory.

---

# Running the Main Script

1. Run the main script using:  
   ```bash
   python main_script.py
   ```

2. **Try Different Settings**  
   - Modify configurations in the `config.yaml` file to adjust feature engineering and model settings.  
   - To run all possible feature engineering combinations, add the `--run_all` option:  
     ```bash
     python main_script.py --run_all
     ```

---

# Result Analysis

1. Analyze the results by running the following Jupyter notebook:  
   ```bash
   output/result_analysis.ipynb
   ```
2. This will generate analysis files in JSON and CSV formats.