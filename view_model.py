from typing import List

from pydantic import BaseModel


class Model(BaseModel):
    model_name : str
    probability: float
    is_toxic: str
