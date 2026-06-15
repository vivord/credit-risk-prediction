import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
from src.utils.config import ARTIFACTS


def get_feature_lists():
    categorical_features = [
        'Status_of_existing_account', 'Credit_history', 'Purpose', 'Savings_account',
        'Present_employment_since', 'Personal_status_and_sex', 'Other_debtors',
        'Property', 'Other_installment_plans', 'Housing', 'Job', 'Telephone', 'foreign_worker'
    ]
    numerical_features = [
        'Duration_in_month', 'Credit_amount', 'Installment_rate_in_percentage_of_disposable_income',
        'Age_in_years', 'Present_residence_since', 'Number_of_existing_credits_at_this_bank',
        'Number_of_people_being_liable_to_provide_maintenance_for'
    ]
    return categorical_features, numerical_features


def get_preprocessor():
    cat_features, num_features = get_feature_lists()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ],
        remainder='drop'  # Drop any extra columns not listed
    )
    return preprocessor


def preprocess_data(df):
    print(f"Original columns: {list(df.columns)}")

    X = df.drop('Risk', axis=1)
    y = df['Risk']

    # Ensure all required columns exist (fill missing with defaults)
    cat_features, num_features = get_feature_lists()
    expected_cols = num_features + cat_features

    for col in expected_cols:
        if col not in X.columns:
            print(f"Warning: Missing column {col} - adding default")
            if col in num_features:
                X[col] = 0
            else:
                X[col] = 'unknown'

    X = X[expected_cols]  # Reorder to exact match

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
    print("✅ Preprocessing completed successfully")

    return X_train_prep, X_test_prep, y_train, y_test, preprocessor, X_train.columns.tolist()