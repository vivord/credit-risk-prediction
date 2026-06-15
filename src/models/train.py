import xgboost as xgb
import joblib
import shap
from sklearn.metrics import roc_auc_score, classification_report
from src.data.ingestion import load_data
from src.features.preprocessing import preprocess_data
from src.utils.config import ARTIFACTS


def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42, eval_metric='auc'
    )
    model.fit(X_train, y_train)

    # Save
    joblib.dump(model, ARTIFACTS / "xgboost_model.joblib")

    # Evaluate
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print("AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])  # Sample
    joblib.dump(explainer, ARTIFACTS / "shap_explainer.joblib")
    return model