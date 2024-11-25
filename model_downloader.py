import os
import subprocess
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

def download_with_aria2(url, destination_path):
    """Скачивание файла с помощью aria2."""
    aria2c_cmd = [
        "aria2c",
        "--file-allocation=none",
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--quiet=false",
        "--dir",
        os.path.dirname(destination_path),
        "--out",
        os.path.basename(destination_path),
        url,
    ]
    print(f"Скачивание {url} в {destination_path}...")
    subprocess.run(aria2c_cmd, check=True)

# Скачивание моделей
for remote_folder, file_list in models_download:
    local_folder = folder_mapping.get(remote_folder, "")
    os.makedirs(local_folder, exist_ok=True)
    for file in file_list:
        destination_path = os.path.join(local_folder, file)
        url = f"{URL_BASE}/{remote_folder}{file}"
        if not os.path.exists(destination_path):
            download_with_aria2(url, destination_path)

# Скачивание индивидуальных файлов
for file_name, local_folder in individual_files:
    os.makedirs(local_folder, exist_ok=True)
    destination_path = os.path.join(local_folder, file_name)
    url = f"{URL_BASE}/{file_name}"
    if not os.path.exists(destination_path):
        download_with_aria2(url, destination_path)

os.system("cls" if os.name == "nt" else "clear")
logger.info("Загрузка завершена успешно.")
