from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from src.utils.config import ARTIFACTS


def get_preprocessor():
    cat_features = ['Status_of_existing_account', 'Credit_history', 'Purpose', ...]  # Add all categorical
    num_features = ['Duration_in_month', 'Credit_amount', 'Age_in_years', ...]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])
    return preprocessor


def preprocess_data(df):
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = get_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
    return X_train_prep, X_test_prep, y_train, y_test, preprocessor