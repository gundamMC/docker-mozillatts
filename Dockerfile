FROM python:3.6 as build

ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get install --yes espeak

RUN mkdir -p /app
RUN cd /app && \
    git clone https://github.com/mozilla/TTS && \
    cd TTS && \
    git checkout c7296b3
    # git checkout 6d6dca0

RUN cd /app/TTS && \
    python3 -m venv .venv

RUN cd /app/TTS && \
    .venv/bin/pip3 install --upgrade pip && \
    .venv/bin/pip3 install -r requirements.txt && \
    .venv/bin/python3 setup.py install && \
    .venv/bin/pip3 install tensorflow==2.3.0rc0

# Extra packages missing from requirements
RUN cd /app/TTS && \
    .venv/bin/pip3 install inflect 'numba==0.48'

# Packages needed for web server
RUN cd /app/TTS && \
    .venv/bin/pip3 install 'flask' 'flask-cors'

# fix 6d6dca0 manually
RUN sed -i 's/import scipy.io/import scipy.io.wavfile/' /app/TTS/utils/audio.py

# -----------------------------------------------------------------------------

FROM python:3.6-slim

RUN apt-get update && \
    apt-get install --yes espeak

COPY --from=build /app /app
COPY vocoder/ /app/vocoder/
COPY model/ /app/model/
COPY templates/ /app/templates/
COPY tts.py scale_stats.npy /app/

WORKDIR /app

EXPOSE 5002

ENTRYPOINT ["/app/TTS/.venv/bin/python3", "/app/tts.py"]