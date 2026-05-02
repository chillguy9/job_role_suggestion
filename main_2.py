from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import numpy as np

app = FastAPI(
    title="Career Prediction API",
    description="Predicts suggested job role based on student skills and preferences.",
    version="1.0.0"
)

# Load model once at startup
model = joblib.load("shubh_model.pkl")


# ── Pydantic input schema ──────────────────────────────────────────────────────

class CareerInput(BaseModel):
    # Numerical features (ratings 1–10)
    logical_quotient_rating: int = Field(..., ge=1, le=10, example=7)
    coding_skills_rating: int = Field(..., ge=1, le=10, example=6)
    hackathons: int = Field(..., ge=0, example=6)
    public_speaking_points: int = Field(..., ge=1, le=10, example=8)

    # Binary yes/no features
    self_learning_capability: Literal["yes", "no"] = Field(..., example="yes")
    extra_courses_did: Literal["yes", "no"] = Field(..., example="no")
    taken_inputs_from_seniors: Literal["yes", "no"] = Field(..., example="yes")
    worked_in_teams: Literal["yes", "no"] = Field(..., example="yes")
    introvert: Literal["yes", "no"] = Field(..., example="no")

    # Ordinal features
    reading_and_writing_skills: Literal["poor", "medium", "excellent"] = Field(..., example="medium")
    memory_capability_score: Literal["poor", "medium", "excellent"] = Field(..., example="excellent")

    # Dummy-encoded features
    management_or_technical: Literal["Management", "Technical"] = Field(..., example="Technical")
    hard_or_smart_worker: Literal["hard worker", "smart worker"] = Field(..., example="smart worker")

    # Category-code features (integer codes — must match training data category ordering)
    interested_subjects_code: int = Field(..., ge=0, example=3)
    interested_type_of_books_code: int = Field(..., ge=0, example=5)
    certifications_code: int = Field(..., ge=0, example=4)
    workshops_code: int = Field(..., ge=0, example=2)
    type_of_company_code: int = Field(..., ge=0, example=1)
    interested_career_area_code: int = Field(..., ge=0, example=0)


# ── Helper: encode input into model feature vector ────────────────────────────

BINARY_MAP = {"yes": 1, "no": 0}
ORDINAL_MAP = {"poor": 0, "medium": 1, "excellent": 2}


def encode(data: CareerInput) -> np.ndarray:
    # Same column order used during training:
    # ['Logical quotient rating', 'coding skills rating', 'hackathons',
    #  'public speaking points', 'self-learning capability?', 'Extra-courses did',
    #  'Taken inputs from seniors or elders', 'worked in teams ever?', 'Introvert',
    #  'reading and writing skills', 'memory capability score',
    #  'B_hard worker', 'B_smart worker', 'A_Management', 'A_Technical',
    #  'Interested subjects_code', 'Interested Type of Books_code',
    #  'certifications_code', 'workshops_code',
    #  'Type of company want to settle in?_code', 'interested career area _code']

    b_hard_worker  = 1 if data.hard_or_smart_worker == "hard worker"  else 0
    b_smart_worker = 1 if data.hard_or_smart_worker == "smart worker" else 0
    a_management   = 1 if data.management_or_technical == "Management" else 0
    a_technical    = 1 if data.management_or_technical == "Technical"  else 0

    features = [
        data.logical_quotient_rating,
        data.coding_skills_rating,
        data.hackathons,
        data.public_speaking_points,
        BINARY_MAP[data.self_learning_capability],
        BINARY_MAP[data.extra_courses_did],
        BINARY_MAP[data.taken_inputs_from_seniors],
        BINARY_MAP[data.worked_in_teams],
        BINARY_MAP[data.introvert],
        ORDINAL_MAP[data.reading_and_writing_skills],
        ORDINAL_MAP[data.memory_capability_score],
        b_hard_worker,
        b_smart_worker,
        a_management,
        a_technical,
        data.interested_subjects_code,
        data.interested_type_of_books_code,
        data.certifications_code,
        data.workshops_code,
        data.type_of_company_code,
        data.interested_career_area_code,
    ]
    return np.array(features).reshape(1, -1)


# ── Response schema ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    predicted_job_role: str
    confidence: float
    all_probabilities: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Career Prediction API is running."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: CareerInput):
    """
    Predicts the suggested job role for a student based on their
    skills, interests, and personality traits.
    """
    try:
        features = encode(data)
        predicted_class = model.predict(features)[0]
        probabilities   = model.predict_proba(features)[0]
        class_labels    = model.classes_

        all_probs = {
            label: round(float(prob), 4)
            for label, prob in zip(class_labels, probabilities)
        }
        confidence = round(float(np.max(probabilities)), 4)

        return PredictionResponse(
            predicted_job_role=str(predicted_class),
            confidence=confidence,
            all_probabilities=all_probs,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
