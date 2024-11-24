import os
import traceback

import librosa
import numpy as np
import av
from io import BytesIO


def wav2(i, o, format):
    inp = av.open(i, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(i, o, format, sr):
    inp = av.open(i, "r")
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    # Устанавливаем параметры до создания потока
    ostream = out.add_stream(format, rate=sr, channels=1)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def load_audio(file, sr):
    file = (
        file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )  # 防止小白拷路径头尾带了空格和"和回车
    if os.path.exists(file) == False:
        raise RuntimeError(
            "You input a wrong audio path that does not exists, please fix it!"
        )
    try:
        with open(file, "rb") as f:
            with BytesIO() as out:
                audio2(f, out, "f32le", sr)
                return np.frombuffer(out.getvalue(), np.float32).flatten()

    except Exception as e:
        # Загружаем аудио с помощью librosa
        audio, _ = librosa.load(file, sr=sr)
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)
        return audio