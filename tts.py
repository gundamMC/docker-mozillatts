#!/usr/bin/env python3
import io
import os
import time
from pathlib import Path

import torch
from flask import Flask, Response, render_template, request
from flask_cors import CORS
from TTS.tf.utils.tflite import load_tflite_model
from TTS.tf.utils.io import load_checkpoint
from TTS.utils.io import load_config
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis

_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------


def run_vocoder(mel_spec):
  vocoder_inputs = mel_spec[None, :, :]
  # get input and output details
  input_details = vocoder_model.get_input_details()
  # reshape input tensor for the new input shape
  vocoder_model.resize_tensor_input(input_details[0]['index'], vocoder_inputs.shape)
  vocoder_model.allocate_tensors()
  detail = input_details[0]
  vocoder_model.set_tensor(detail['index'], vocoder_inputs)
  # run the model
  vocoder_model.invoke()
  # collect outputs
  output_details = vocoder_model.get_output_details()
  waveform = vocoder_model.get_tensor(output_details[0]['index'])
  return waveform 


def tts(model, text, CONFIG, p):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, style_wav=None,
                                                                             truncated=False, enable_eos_bos_chars=CONFIG.enable_eos_bos_chars,
                                                                             backend='tflite')
    waveform = run_vocoder(mel_postnet_spec.T)
    waveform = waveform[0, 0]
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(waveform.shape)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    return alignment, mel_postnet_spec, stop_tokens, waveform


# -----------------------------------------------------------------------------

# runtime settings
use_cuda = False

# model paths
TTS_MODEL = str(_DIR / "model" / "tts_model.tflite")
TTS_CONFIG = str(_DIR / "model" / "config.json")
VOCODER_MODEL = str(_DIR / "vocoder" / "vocoder_model.tflite")
VOCODER_CONFIG = str(_DIR / "vocoder" / "config_vocoder.json")

# load configs
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

# load the audio processor
ap = AudioProcessor(**TTS_CONFIG.audio)

# LOAD TTS MODEL
# multi speaker
speaker_id = None
speakers = []

# load the model
model = load_tflite_model(TTS_MODEL)
vocoder_model = load_tflite_model(VOCODER_MODEL)

# -----------------------------------------------------------------------------

app = Flask("mozillatts")
CORS(app)

# -----------------------------------------------------------------------------


@app.route("/api/tts")
def api_tts():
    text = request.args.get("text", "").strip()
    align, spec, stop_tokens, wav = tts(model, text, TTS_CONFIG, ap)

    with io.BytesIO() as out:
        ap.save_wav(wav, out)
        return Response(out.getvalue(), mimetype="audio/wav")


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
