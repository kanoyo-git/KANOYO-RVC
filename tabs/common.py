import os
import sys
from dotenv import load_dotenv
import requests
import subprocess
import wave
import zipfile
from mega import Mega
from urllib.parse import urlencode
import shutil
import logging
import soundfile as sf
import numpy as np
import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pathlib
import json
from pydub import AudioSegment
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import edge_tts, asyncio
import torch
from i18n.i18n import I18nAuto
from configs.config import Config

# Настройка логгера
logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Инициализация рабочей директории
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

# Создание временных директорий
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % now_dir, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "models/pth"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

# Загрузка конфигурации
config = Config()

# Импорт необходимых модулей для работы приложения
from infer.modules.vc.modules import VC
from infer.modules.vc.ilariatts import tts_order_voice
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from tools import pretrain_helper

# Инициализация i18n и VC
i18n = I18nAuto()
vc = VC(config)

# Загрузка путей и перечней файлов
weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
audio_root = "audios"
sup_audioext = {'wav', 'mp3', 'flac', 'ogg', 'opus',
                'm4a', 'mp4', 'aac', 'alac', 'wma',
                'aiff', 'webm', 'ac3'}

# Получение списков моделей и индексов
names = [os.path.join(root, file)
         for root, _, files in os.walk(weight_root)
         for file in files
         if file.endswith((".pth", ".onnx"))]

indexes_list = [os.path.join(root, name)
               for root, _, files in os.walk(index_root, topdown=False) 
               for name in files 
               if name.endswith(".index") and "trained" not in name]

audio_paths = [os.path.join(root, name)
               for root, _, files in os.walk(audio_root, topdown=False) 
               for name in files
               if name.endswith(tuple(sup_audioext))]

audio_paths = [str(path) for path in audio_paths]

# Получение списка голосов для TTS
language_dict = tts_order_voice
ilariavoices = list(language_dict.keys())

# Определение словаря семплрейтов
sr_dict = {
    "32k": 32000, "40k": 40000, "48k": 48000, "OV2-32k": 32000, "OV2-40k": 40000, 
    "RIN-40k": 40000, "Snowie-40k": 40000, "Snowie-48k": 48000, 
    "SnowieV3.1-40k": 40000, "SnowieV3.1-32k": 32000, "SnowieV3.1-48k": 48000, 
    "SnowieV3.1-RinE3-40K": 40000, "Italia-32k": 32000,
}

# Вспомогательные функции
def get_pretrained_files(directory, keyword, filter_str):
    file_paths = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pth") and keyword in filename and filter_str in filename:
            file_paths[filename] = os.path.join(directory, filename)
    return file_paths

# Получение предобученных моделей
pretrained_directory = "assets/pretrained_v2"
pretrained_path = {filename: os.path.join(pretrained_directory, filename) 
                  for filename in os.listdir(pretrained_directory)}
pretrained_G_files = get_pretrained_files(pretrained_directory, "G", "f0")
pretrained_D_files = get_pretrained_files(pretrained_directory, "D", "f0")

# Проверка GPU
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10", "16", "20", "30", "40", "A2", "A3", "A4", "P4",
                "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN",
            ]
        ):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024 / 1024 / 1024 + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = ((min(mem) // 2 + 1) // 2) * 2
else:
    gpu_info = i18n("Your GPU doesn't work for training")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

F0GPUVisible = config.dml is False

def get_pretrained_models(path_str, f0_str, sr2):
    sr_mapping = pretrain_helper.get_pretrained_models(f0_str)

    pretrained_G_filename = sr_mapping.get(sr2, "")
    pretrained_D_filename = pretrained_G_filename.replace("G", "D")

    if not pretrained_G_filename or not pretrained_D_filename:
        logging.warning(f"Pretrained models not found for sample rate {sr2}, will not use pretrained models")

    return os.path.join(pretrained_directory, pretrained_G_filename), os.path.join(pretrained_directory, pretrained_D_filename)

def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    audio_paths = [os.path.join(audio_root, file) for file in os.listdir(os.path.join(now_dir, "audios"))]

    return {
        "choices": sorted(names), "__type__": "update"
    }, {
        "choices": sorted(index_paths), "__type__": "update"
    }, {
        "choices": sorted(audio_paths), "__type__": "update"
    }

def clean():
    return {"value": "", "__type__": "update"}

def generate_spectrogram_and_get_info(audio_file):
    y, sr = librosa.load(audio_file, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    log_S = librosa.amplitude_to_db(S, ref=np.max, top_db=256)

    plt.figure(figsize=(12, 5.5))
    librosa.display.specshow(log_S, sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB', pad=0.01)
    plt.tight_layout(pad=0.5)

    plt.savefig('spectrogram.png', dpi=500)

    audio_info = sf.info(audio_file)
    bit_depth = {'PCM_16': 16, 'FLOAT': 32}.get(audio_info.subtype, 0)
    minutes, seconds = divmod(audio_info.duration, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds *= 1000
    speed_in_kbps = audio_info.samplerate * bit_depth / 1000
    filename_without_extension, _ = os.path.splitext(os.path.basename(audio_file))

    info_table = f"""
    | Information | Value |
    | :---: | :---: |
    | File Name | {filename_without_extension} |
    | Duration | {int(minutes)} minutes - {int(seconds)} seconds - {int(milliseconds)} milliseconds |
    | Bitrate | {speed_in_kbps} kbp/s |
    | Audio Channels | {audio_info.channels} |
    | Samples per second | {audio_info.samplerate} Hz |
    | Bit per second | {audio_info.samplerate * audio_info.channels * bit_depth} bit/s |
    """

    return info_table, "spectrogram.png"

def get_audio_duration(audio_file_path):
    audio_info = sf.info(audio_file_path)
    duration_minutes = audio_info.duration / 60
    return duration_minutes

def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True

def if_done_multi(done, ps):
    while 1:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

def import_files(file):
    if file is not None:
        file_name = file.name
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                # Create a temporary directory to extract files
                temp_dir = './TEMP'
                zip_ref.extractall(temp_dir)
                # Move .pth and .index files to their respective directories
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.pth'):
                            destination = './models/pth/' + file
                            if not os.path.exists(destination):
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"File {destination} already exists. Skipping.")
                        elif file.endswith('.index'):
                            destination = './models/index/' + file
                            if not os.path.exists(destination):
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"File {destination} already exists. Skipping.")
                # Remove the temporary directory
                shutil.rmtree(temp_dir)
            return "Zip file has been successfully extracted."
        elif file_name.endswith('.pth'):
            destination = './models/pth/' + os.path.basename(file.name)
            if not os.path.exists(destination):
                os.rename(file.name, destination)
            else:
                print(f"File {destination} already exists. Skipping.")
            return "PTH file has been successfully imported."
        elif file_name.endswith('.index'):
            destination = './models/index/' + os.path.basename(file.name)
            if not os.path.exists(destination):
                os.rename(file.name, destination)
            else:
                print(f"File {destination} already exists. Skipping.")
            return "Index file has been successfully imported."
        else:
            return "Unsupported file type."
    else:
        return "No file has been uploaded."

def download_from_url(url, model):
    if url == '':
        return "URL cannot be left empty."
    if model == '':
        return "You need to name your model. For example: Ilaria"

    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)

    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)

    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile

    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        elif "disk.yandex.ru" in url:
            base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
            public_key = url
            final_url = base_url + urlencode(dict(public_key=public_key))
            response = requests.get(final_url)
            download_url = response.json()['href']
            download_response = requests.get(download_url)
            with open(zipfile_path, 'wb') as file:
                file.write(download_response.content)
        else:
            response = requests.get(url)
            response.raise_for_status() 
            with open(zipfile_path, 'wb') as file:
                file.write(response.content)

        shutil.unpack_archive(zipfile_path, "./unzips", 'zip')

        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.makedirs(f'./models/index', exist_ok=True)
                    shutil.copy2(file_path, f'./models/index/{model}.index')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    os.makedirs(f'./models/pth', exist_ok=True)
                    shutil.copy(file_path, f'./models/pth/{model}.pth')

        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return i18n("Model downloaded, you can go back to the inference page!")

    except subprocess.CalledProcessError as e:
        return f"ERROR - Download failed (gdown): {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"ERROR - Download failed (requests): {str(e)}"
    except Exception as e:
        return f"ERROR - The test failed: {str(e)}"

def calculate_remaining_time(epochs, seconds_per_epoch):
    total_seconds = epochs * seconds_per_epoch

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours == 0:
        return f"{int(minutes)} minutes"
    elif hours == 1:
        return f"{int(hours)} hour and {int(minutes)} minutes"
    else:
        return f"{int(hours)} hours and {int(minutes)} minutes"

# Обновление CSS для Gradio 5.x
css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
}
.output-image, .input-image {
    height: auto !important;
}
.gr-form {
    flex-grow: 1;
}
.gr-box {
    border-radius: 8px;
    padding: 15px;
}
.gr-padded {
    padding: 16px;
}
.gr-input, .gr-dropdown, .gr-textbox, .gr-textarea {
    box-shadow: none !important;
}
.gr-slider {
    padding: 8px 0;
}
.gr-button {
    border-radius: 6px;
}
"""

# Обновление функции для создания интерфейсов
def create_ui_element(element_type, *args, **kwargs):
    """Обёртка для создания элементов UI с корректными размерами для Gradio 5.x"""
    # Добавляем стандартные размеры для элементов
    if 'elem_classes' not in kwargs:
        kwargs['elem_classes'] = []
    
    # Настройка для специфичных элементов
    if element_type == gr.Textbox:
        if 'lines' not in kwargs:
            kwargs['lines'] = 1
        if 'scale' not in kwargs:
            kwargs['scale'] = 1
    
    # Возвращаем созданный элемент
    return element_type(*args, **kwargs) 