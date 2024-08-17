import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from nisqa.model import NisqaModel

app = FastAPI()

pretrained_model = os.path.join(os.getcwd(), "weights", "nisqa.tar")
device = "cuda:1"


@app.post("/predict")
async def predict(
    audio_file: UploadFile = File(...),
    sr: int = Form(None),
):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await audio_file.read())
        temp_path = temp_file.name

    args = {
        "pretrained_model": pretrained_model,
        "filename": temp_path,
        "ms_channel": None,
        "ms_sr": sr,
        "run_device": device,
    }

    nisqa = NisqaModel(args)

    scores = nisqa.predict(temp_path)

    os.unlink(temp_path)

    return scores


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8356)
