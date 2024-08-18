from glob import glob
from pathlib import Path

from nisqa.model import NisqaModel


pretrained_model = Path.cwd() / "weights" / "nisqa.tar"
device = "cuda:1"


nisqa = NisqaModel(
    {
        "pretrained_model": pretrained_model,
        "ms_channel": None,
        "ms_sr": 16_000,
        "device": device,
    }
)

filenames = glob("test_audios/*.wav")

for filename in filenames:
    scores = nisqa.predict(filename)

    print(scores)
