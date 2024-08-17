import os
import logging
import tempfile

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from nisqa.model import NisqaModel

app = FastAPI()

current_dir = os.getcwd()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/predict")
async def predict(
    audio_file: UploadFile = File(...),
    pretrained_model: str = Form(...),
    ms_sr: int = Form(None),
    ms_channel: int = Form(None),
):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await audio_file.read())
        temp_path = temp_file.name

    args = {
        "pretrained_model": os.path.join(current_dir, "weights", pretrained_model),
        "deg": temp_path,
        "ms_channel": ms_channel,
        "ms_sr": ms_sr,
        "tr_bs_val": 1,
        "tr_num_workers": 0,
        "run_device": "cuda:1",
    }

    try:
        nisqa = NisqaModel(args)

        scores = nisqa.predict()

        return {
            "mos": float(scores["mos_pred"]),
            "noi": float(scores["noi_pred"]),
            "dis": float(scores["dis_pred"]),
            "col": float(scores["col_pred"]),
            "loud": float(scores["loud_pred"]),
        }
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8356)
