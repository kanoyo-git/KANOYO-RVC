import os
import requests
from tqdm import tqdm
import subprocess
import shutil
import platform
import logging

logger = logging.getLogger(__name__)

URL_BASE = "https://huggingface.co/kanoyo/0v2Super/resolve/main"
models_download = [
    (
        "pretrained_v2/",
        [
            "D_Snowie-X-Rin_40k.pth",
            "D_SnowieV3.1_40k.pth",
            "D_SnowieV3.1_48k.pth",
            "G_Snowie-X-Rin_40k.pth",
            "G_SnowieV3.1_40k.pth",
            "G_SnowieV3.1_48k.pth",
            "f0Ov2Super40kD.pth",
            "f0Ov2Super40kG.pth",
        ],
    ),
    ("", ["", ""]),
]


individual_files = [
    ("hubert_base.pt", "assets/hubert/"),
    ("rmvpe.pt", "assets/rmvpe/"),
    ("rmvpe.onnx", "assets/rmvpe/"),
]

folder_mapping = {
    "pretrained_v2/": "assets/pretrained_v2/",
    "": "",
}


def download_file_with_progress(url, destination_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with open(destination_path, "wb") as file, tqdm(
        desc=os.path.basename(destination_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))


if not os.path.exists("torchcrepe"):
    os_name = platform.system()

    print("Клонирование репозитория GitHub во временную директорию...")

    mingit_path = os.path.join(os.getcwd(), "lib", "tools", "mingit", "cmd", "git.exe")

    if os.path.exists(mingit_path):
        subprocess.run(
            [
                mingit_path,
                "clone",
                "https://github.com/maxrmorrison/torchcrepe.git",
                "temp_torchcrepe",
            ]
        )
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/maxrmorrison/torchcrepe.git",
                "temp_torchcrepe",
            ]
        )

    print("Копирование папки torchcrepe...")
    shutil.copytree("temp_torchcrepe/torchcrepe", "./torchcrepe")

    print("Удаление временной директории...")
    print(os_name)
    if os_name == "Windows":
        subprocess.run("rmdir /s /q temp_torchcrepe", shell=True)
    if os_name == "Linux":
        shutil.rmtree("temp_torchcrepe")

for remote_folder, file_list in models_download:
    local_folder = folder_mapping.get(remote_folder, "")
    for file in file_list:
        destination_path = os.path.join(local_folder, file)
        url = f"{URL_BASE}/{remote_folder}{file}"
        if not os.path.exists(destination_path):
            print(f"Скачивание {url} в {destination_path}...")
            download_file_with_progress(url, destination_path)

for file_name, local_folder in individual_files:
    destination_path = os.path.join(local_folder, file_name)
    url = f"{URL_BASE}/{file_name}"
    if not os.path.exists(destination_path):
        print(f"Скачивание {url} в {destination_path}...")
        download_file_with_progress(url, destination_path)

os.system("cls" if os.name == "nt" else "clear")
logger.info("Загрузка Kanoyo успешно продолжается...")
