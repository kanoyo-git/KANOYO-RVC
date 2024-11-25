import os
import subprocess
import argparse
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
    if os.path.exists(destination_path):
        print(f"Файл {destination_path} уже существует. Пропуск.")
        return
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

def download_files(file_list, folder_mapping, url_base):
    for remote_folder, file_list in file_list:
        local_folder = folder_mapping.get(remote_folder, "")
        os.makedirs(local_folder, exist_ok=True)
        for file in file_list:
            destination_path = os.path.join(local_folder, file)
            url = f"{url_base}/{remote_folder}{file}"
            download_with_aria2(url, destination_path)

def main():
    parser = argparse.ArgumentParser(description="Скачивание моделей и ресурсов.")
    parser.add_argument(
        "-train",
        action="store_true",
        help="Скачивать претрейны вместе с главными файлами.",
    )
    args = parser.parse_args()

    if args.train:
        print("Режим тренировки включен. Скачиваются претрейны...")
        download_files(models_download, folder_mapping, URL_BASE)
        for file_name, local_folder in individual_files:
            os.makedirs(local_folder, exist_ok=True)
            destination_path = os.path.join(local_folder, file_name)
            url = f"{URL_BASE}/{file_name}"
            download_with_aria2(url, destination_path)
    else:
        print("Режим без тренировки. Скачиваются только главные файлы...")
        for file_name, local_folder in individual_files:
            os.makedirs(local_folder, exist_ok=True)
            destination_path = os.path.join(local_folder, file_name)
            url = f"{URL_BASE}/{file_name}"
            download_with_aria2(url, destination_path)

    logger.info("Скачивание завершено.")

if __name__ == "__main__":
    main()
