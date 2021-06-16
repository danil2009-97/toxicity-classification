from typing import List


from pydantic import BaseModel, Field


class Model(BaseModel):
    model_name : str
    probability: float
    is_toxic: str
    prediction_time : str = Field("")

