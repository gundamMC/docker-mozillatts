# Mozilla TTS-TFLite

Docker image for [Mozilla TTS](https://github.com/mozilla/TTS) with Tensorflow Lite support based on [@erogol's](https://github.com/erogol) [TFLite example](https://colab.research.google.com/drive/1bcTcFTTbws8l8gpCITN_18gkK5-GHMiY?usp=sharing) and edited from [synesthesiam/docker-mozillatts](https://github.com/synesthesiam/docker-mozillatts).

Includes TFLite-optimized versions of [@erogol's](https://github.com/erogol) pre-built LJSpeech Tacotron2 English model and Multiband MelGAN vocoder.
See [below](#building-yourself) for links to specific checkpoints.

## Using

```sh
$ docker build -t tts .
$ docker run -it -p 5002:5002 tts
```

Visit http://localhost:5002 for web interface.

Do HTTP GET at http://localhost:5002/api/tts?text=your%20sentence to get WAV audio back:

```sh
$ curl -G --output - \
    --data-urlencode 'text=Welcome to the world of speech synthesis!' \
    'http://localhost:5002/api/tts' | \
    aplay
```

## Building Yourself

The Docker image is built using [the TFLite Example](https://colab.research.google.com/drive/1bcTcFTTbws8l8gpCITN_18gkK5-GHMiY?usp=sharing). You'll need to manually download the model and vocoder checkpoints/configs:

* [`model/tts_model.tflite`](https://drive.google.com/uc?id=17PYXCmTe0el_SLTwznrt3vOArNGMGo5v)
* [`model/config.json`](https://drive.google.com/uc?id=18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc)
* [`vocoder/vocoder_model.tflite`](https://drive.google.com/uc?id=1aXveT-NjOM1mUr6tM4JfWjshq67GvVIO)
* [`vocoder/config_vocoder.json`](https://drive.google.com/uc?id=1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu)
* [`scale_stats.npy`](https://drive.google.com/uc?id=11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU)
