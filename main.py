"""main.py - FastAPI app.   Local:  python -m uvicorn main:app --reload"""
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import engine

app = FastAPI(title="Student Early-Warning API",
              description="Predicts Dropout / Enrolled / Graduate from admission, semester-1 and semester-2 data "
                          "(Logistic Regression, UCI 697, nested-CV validated).", version="1.0")


class Semester(BaseModel):
    registered: int = Field(..., ge=0, le=40, description="Subjects registered")
    passed: int = Field(..., ge=0, le=40, description="Subjects passed")
    sgpa: float = Field(..., ge=0, le=10, description="SGPA on a 0-10 scale (0 if nothing was evaluated)")

    @model_validator(mode="after")
    def passed_not_more_than_registered(self):
        if self.passed > self.registered:
            raise ValueError("passed cannot be greater than registered")
        return self


class Student(BaseModel):
    previous_qualification_percent: float = Field(..., ge=0, le=100, description="12th / previous qualification %")
    admission_percent: Optional[float] = Field(None, ge=0, le=100, description="Admission score %, leave out if unknown")
    hostel: int = Field(..., ge=0, le=1, description="1 = studying away from home / hostel")
    first_generation: int = Field(..., ge=0, le=1, description="1 = neither parent attended higher education")
    scholarship: int = Field(..., ge=0, le=1)
    fee_dues: int = Field(0, ge=0, le=1, description="1 = fees pending (used from semester 1 onwards)")
    semester1: Optional[Semester] = Field(None, description="Leave out at admission time (checkpoint A)")
    semester2: Optional[Semester] = Field(None, description="Add after semester 2 (checkpoint C)")

    model_config = {"json_schema_extra": {"example": {
        "previous_qualification_percent": 72, "admission_percent": 68, "hostel": 1, "first_generation": 1,
        "scholarship": 0, "fee_dues": 1,
        "semester1": {"registered": 6, "passed": 3, "sgpa": 5.5},
        "semester2": {"registered": 6, "passed": 2, "sgpa": 4.8}}}}


@app.get("/")
def root():
    return {"message": "Student Early-Warning API is running. Open /docs to test.", "endpoints": ["/health", "/predict", "/docs"]}


@app.get("/health")
def health():
    return {"status": "ok", "checkpoints_loaded": list(engine.BUNDLE["models"])}


@app.post("/predict")
def predict(student: Student):
    try:
        return engine.predict(student.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
