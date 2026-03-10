python
from functools import lru_cache

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "bodyPerformance.csv"
RENAME_MAP = {
    "body fat_%": "body_fat",
    "gripForce": "grip_force",
    "sit and bend forward_cm": "sit_bend_forward",
    "sit-ups counts": "sit_ups",
    "broad jump_cm": "broad_jump",
    "class": "target"
}

FEATURE_COLS = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "body_fat",
    "diastolic",
    "systolic",
    "grip_force",
    "sit_bend_forward",
    "sit_ups",
    "broad_jump"
]

TARGET_COL = "target"

TARGET_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}

REVERSE_TARGET_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="Fitness Grade Predictor API",
    version="1.0.0",
    description="API for predicting fitness class from body performance data"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # للتجربة فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request / Response Models
# ---------------------------
class PredictRequest(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: str = Field(..., pattern="^(F|M|f|m)$")
    height_cm: float = Field(..., ge=100, le=250)
    weight_kg: float = Field(..., ge=30, le=250)
    body_fat: float = Field(..., ge=0, le=70)
    diastolic: float = Field(..., ge=30, le=200)
    systolic: float = Field(..., ge=50, le=300)
    grip_force: float = Field(..., ge=0, le=150)
    sit_bend_forward: float = Field(..., ge=-100, le=150)
    sit_ups: float = Field(..., ge=0, le=300)
    broad_jump: float = Field(..., ge=0, le=500)


class PredictResponse(BaseModel):
    predicted_class: str
    class_meaning: str
    top_feature: str
    explanation: str
    recommendation: list[str]
    model_accuracy: float
    model_f1: float


# ---------------------------
# Helpers
# ---------------------------
def class_meaning(label: str) -> str:
    meanings = {
        "A": "Excellent fitness level",
        "B": "Good fitness level",
        "C": "Average fitness level",
        "D": "Needs improvement"
    }
    return meanings.get(label, "Unknown")


def generate_fitness_recommendation(predicted_class: str, top_feature: str) -> list[str]:
    return [
        f"Follow a consistent weekly exercise routine suitable for class {predicted_class}.",
        f"Pay extra attention to improving {top_feature} through focused training.",
        "Track your progress every 2 to 4 weeks using the same fitness measurements."
    ]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns=RENAME_MAP)
    return df


def preprocess_data(df: pd.DataFrame):
    if df is None:
        raise ValueError("Dataset is None.")

    df = df.copy()
    df = df.rename(columns=RENAME_MAP)

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    gender_series = df["gender"].astype(str).str.strip().str.upper()
    gender_map = {
        "F": 0,
        "M": 1,
        "0": 0,
        "1": 1
    }
    df["gender"] = gender_series.map(gender_map)

    target_series = df[TARGET_COL].astype(str).str.strip().str.upper()
    y = target_series.map(TARGET_MAP)

    X = df[FEATURE_COLS].copy()

    if X.isnull().any().any():
        null_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Some feature values could not be encoded correctly. Problem columns: {null_cols}")

    if y.isnull().any():
        bad_vals = sorted(df[TARGET_COL].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unexpected target values found: {bad_vals}")

    return X, y


@lru_cache
def get_model_bundle():
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, acc, f1, importance_df


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {"message": "Fitness Grade Predictor API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    try:
        model, acc, f1, importance_df = get_model_bundle()

        input_df = pd.DataFrame([{
            "age": data.age,
            "gender": 1 if data.gender.strip().upper() == "M" else 0,
            "height_cm": data.height_cm,
            "weight_kg": data.weight_kg,
            "body_fat": data.body_fat,
            "diastolic": data.diastolic,
            "systolic": data.systolic,
            "grip_force": data.grip_force,
            "sit_bend_forward": data.sit_bend_forward,
            "sit_ups": data.sit_ups,
            "broad_jump": data.broad_jump
        }])

        pred_num = int(model.predict(input_df)[0])
        pred_label = REVERSE_TARGET_MAP[pred_num]

        top_feature = str(importance_df.iloc[0]["Feature"])

        explanation = (
            f"The model predicted class {pred_label}. "
            f"The strongest global signal in the model is currently {top_feature}."
        )

        recommendation = generate_fitness_recommendation(
            predicted_class=pred_label,
            top_feature=top_feature
        )

        return PredictResponse(
            predicted_class=pred_label,
            class_meaning=class_meaning(pred_label),
            top_feature=top_feature,
            explanation=explanation,
            recommendation=recommendation,
            model_accuracy=round(float(acc), 4),
            model_f1=round(float(f1), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

