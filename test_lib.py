from glob import glob
from pathlib import Path

from nisqa.model import NisqaModel


pretrained_model = Path.cwd() / "weights" / "nisqa.tar"
device = "cuda:1"

filenames = glob("test_audios/*.wav")

for filename in filenames:
    args = {
        "pretrained_model": pretrained_model,
        "ms_channel": None,
        "ms_sr": 16_000,
        "device": device,
    }

    nisqa = NisqaModel(args)
    scores = nisqa.predict(filename)

    print(scores)
