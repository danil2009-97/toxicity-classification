from load_model import *
from fastapi import FastAPI
from view_model import *
import uvicorn
from make_prediction import *
from typing import List


app = FastAPI()


@app.get("/get-prediction", response_model= List[Model])
def get_model_prediction(text : str = """stop being so bad as hell""") -> List[Model]:
    result = get_prediction(text)

    return result


if __name__ == '__main__':
    uvicorn.run(app)

