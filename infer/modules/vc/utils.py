import os
import torch
import logging
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary

# Добавляем Dictionary в список безопасных глобальных объектов для PyTorch 2.6+
torch.serialization.add_safe_globals([Dictionary])

logger = logging.getLogger(__name__)

def get_index_path_from_model(sid):
    # Извлекаем базовое имя модели без расширения и пути
    model_name = os.path.basename(sid).split(".")[0]
    logger.info(f"Ищем индекс для модели: {model_name}")
    
    # Получаем все индексы из директории index_root
    index_root = os.getenv("index_root")
    if not index_root or not os.path.exists(index_root):
        logger.warning(f"Директория индексов не существует: {index_root}")
        return ""
    
    all_indexes = []
    for root, _, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_path = os.path.join(root, name)
                all_indexes.append(index_path)
                
    logger.info(f"Найдено {len(all_indexes)} индексов")
    
    # Ищем индекс, соответствующий имени модели
    for index_path in all_indexes:
        index_name = os.path.basename(index_path)
        if model_name in index_name or model_name in index_path:
            logger.info(f"Выбран индекс: {index_path}")
            return index_path
    
    # Если точного совпадения нет, пробуем искать частичное совпадение
    for index_path in all_indexes:
        if model_name.lower() in index_path.lower():
            logger.info(f"Выбран индекс по частичному совпадению: {index_path}")
            return index_path
    
    logger.warning(f"Индекс для модели {model_name} не найден")
    return ""


def load_hubert(config):
    try:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["assets/hubert/hubert_base.pt"],
            suffix="",
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(config.device)
        if config.is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        return hubert_model.eval()
    except Exception as e:
        # Альтернативный метод загрузки с weights_only=False если основной метод не сработал
        if "weights_only" in str(e) or "UnpicklingError" in str(e):
            print("Пробуем альтернативный метод загрузки hubert модели с weights_only=False")
            
            # Модифицируем функцию загрузки чекпоинта в checkpoint_utils
            original_load_checkpoint = checkpoint_utils.load_checkpoint_to_cpu
            
            def patched_load_checkpoint(filename, *args, **kwargs):
                with open(filename, "rb") as f:
                    state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
                return state
            
            # Подменяем функцию на время загрузки
            checkpoint_utils.load_checkpoint_to_cpu = patched_load_checkpoint
            
            try:
                models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                    ["assets/hubert/hubert_base.pt"],
                    suffix="",
                )
                hubert_model = models[0]
                hubert_model = hubert_model.to(config.device)
                if config.is_half:
                    hubert_model = hubert_model.half()
                else:
                    hubert_model = hubert_model.float()
                return hubert_model.eval()
            finally:
                # Восстанавливаем оригинальную функцию
                checkpoint_utils.load_checkpoint_to_cpu = original_load_checkpoint
        else:
            # Если ошибка другая, просто передаем ее дальше
            raise e
