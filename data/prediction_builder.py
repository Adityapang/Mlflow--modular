import pandas as pd
from datetime import datetime
from config import DATE_COL, TARGET_COL, DISTRICT_COL

def build_prediction_data(predictions: pd.Series,
                          X_test_meta: pd.DataFrame,
                          best_cv_rmse: float,
                          test_rmse: float) -> pd.DataFrame:
    prediction_df = pd.DataFrame({
        DATE_COL: X_test_meta[DATE_COL],
        'predicted_case_count': predictions,
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        'best_cv_rmse': best_cv_rmse,
        "test_rmse": test_rmse,
        DISTRICT_COL: X_test_meta[DISTRICT_COL],
        TARGET_COL: X_test_meta[TARGET_COL]
    })
    return prediction_df