from RealtimeSTT import AudioToTextRecorder
import ollama
import pyaudio
import wave
# import sys
# from melo.api import TTS
import subprocess

MODEL_NAME = "samantha-mistral"
DEVICE = "cuda:0"
CHUNK = 1024

# p = pyaudio.PyAudio()

recorder = AudioToTextRecorder(model="base.en", language="en", spinner=True,
                               debug_mode=False,
                               enable_realtime_transcription=True, realtime_model_type="tiny.en",
                               use_microphone=True,
                               input_device_index=15
                               )

full_sentences = []

# model = TTS(language="EN")
# speaker_ids = model.hps.data.spk2id
output_path = "/home/optimuseprime/Projects/MeloTTS/1_output.wav"

p = pyaudio.PyAudio()

def process_text(txt):
    full_sentences.append(txt)
    print(full_sentences)
    resp = ollama.generate(model=MODEL_NAME, prompt=txt, stream=False)
    print(resp)
    # model.tts_to_file(resp["response"], speaker_ids["EN-US"], output_path, speed=1.0)
    subprocess.run(["melo", f"'{resp['response']}'", "--language", "EN", "--speaker", "EN-US", "1_output.wav"], cwd="/home/optimuseprime/Projects/MeloTTS")
    wf = wave.open(output_path, 'rb')

    # open stream based on the wave object which has been input.
    stream = p.open(format= p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)

    # read data (based on the chunk size)
    data = wf.readframes(CHUNK)

    # play stream (looping from beginning of file to the end)
    while data:
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(CHUNK)


while True:
    recorder.text(process_text)
