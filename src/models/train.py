import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("✅ Starting training script...")

try:
    from src.data.ingestion import load_data
    from src.features.preprocessing import preprocess_data
    from src.utils.config import ARTIFACTS

    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

import joblib
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report


def train_model():
    print("🚀 Starting Credit Risk Training Pipeline...")
    try:
        print("Loading data...")
        df = load_data()
        print(f"✅ Dataset shape: {df.shape}")

        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
        print("✅ Preprocessing completed")

        print("Training XGBoost model...")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(
                y_train[y_train == 1]) > 0 else 1,
            random_state=42,
            eval_metric='auc'
        )
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, ARTIFACTS / "xgboost_model.joblib")

        # Evaluate
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"✅ AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, preds))

        # SHAP Explainer
        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, ARTIFACTS / "shap_explainer.joblib")
        joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")  # Ensure saved

        print("🎉 Model training completed successfully!")
        return model, feature_names

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    train_model()