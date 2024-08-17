import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
        "deg": temp_path,
        "ms_channel": None,
        "ms_sr": sr,
        "tr_bs_val": 1,
        "tr_num_workers": 0,
        "run_device": device,
    }

    try:
        nisqa = NisqaModel(args)

        scores = nisqa.predict()

        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8356)
