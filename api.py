import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from nisqa.model import NisqaModel

app = FastAPI()

pretrained_model = Path.cwd() / "weights" / "nisqa.tar"
device = "cuda:1"

args = {
    "pretrained_model": pretrained_model,
    "ms_channel": None,
    "ms_sr": 16_000,
    "device": device,
}

nisqa = NisqaModel(args)


@app.post("/predict")
async def predict(
    audio_file: UploadFile = File(...),
    sr: int = Form(None),
):
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(await audio_file.read())
        temp_path = temp_file.name

        return nisqa.predict(temp_path, sr=sr)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8356)
