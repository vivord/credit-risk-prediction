import pandas as pd
import urllib.request
from src.utils.config import DATA_RAW

def download_german_credit():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    filepath = DATA_RAW / "german.data"
    if not filepath.exists():
        urllib.request.urlretrieve(url, filepath)
    return filepath

def load_data():
    col_names = [
        'Status_of_existing_account', 'Duration_in_month', 'Credit_history', 'Purpose',
        'Credit_amount', 'Savings_account', 'Present_employment_since', 'Installment_rate',
        'Personal_status_and_sex', 'Other_debtors', 'Present_residence_since', 'Property',
        'Age_in_years', 'Other_installment_plans', 'Housing', 'Number_of_existing_credits',
        'Job', 'Number_of_people_being_liable', 'Telephone', 'foreign_worker', 'Risk'
    ]
    df = pd.read_csv(download_german_credit(), sep=' ', header=None, names=col_names)
    df['Risk'] = df['Risk'].map({1: 0, 2: 1})  # 0=Good, 1=Bad (default)
    return df