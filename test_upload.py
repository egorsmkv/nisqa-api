import requests

endpoint_url = "http://127.0.0.1:8356/predict"

# Test 1

filename = "test_audios/1.wav"
form_data = {"pretrained_model": "nisqa.tar", "ms_sr": 16000, "ms_channel": 1}
response = requests.post(
    endpoint_url, files={"audio_file": open(filename, "rb")}, data=form_data
)

print(response.json())

# Test 2

filename = "test_audios/2.wav"
form_data = {"pretrained_model": "nisqa.tar", "ms_sr": 16000, "ms_channel": 1}
response = requests.post(
    endpoint_url, files={"audio_file": open(filename, "rb")}, data=form_data
)

print(response.json())
