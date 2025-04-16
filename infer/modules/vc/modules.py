import traceback
import logging
import tempfile

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO
import hashlib

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[0]
            if self.if_f0 != 0 and to_return_protect
            else 0.5,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[1]
            if self.if_f0 != 0 and to_return_protect
            else 0.33,
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if self.hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = (
                    self.net_g
                ) = self.n_spk = self.hubert_model = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")
        self.eps = self.cpt["info"]
        if(self.eps==""):
            self.eps="N/A"

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        model_hash = hashlib.md5(open(person, "rb").read()).hexdigest()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        if self.eps == 'N/A':
            epoch_str = "This is a v1 model, on some older models epochs were not used."
        else:
            epoch_str = self.eps[:-5] if isinstance(self.eps, str) and self.eps.endswith('epoch') else str(self.eps)

        fstr = f"Epochs: {epoch_str}\nSample Rate: {self.tgt_sr}\nVersion: {self.version}\nHash: {model_hash}"
        logger.info(fstr)

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
                fstr,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path_uploaded,
        input_audio_path_select,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        # if the type of input_audio_path is temporaryFileWrapper, it means the user has used a voice recording, so use the filepath of the temporary file
        if (input_audio_path_uploaded is None) and (input_audio_path_select is None):
            return "Необходимо загрузить аудио или выбрать из списка", None

        if sid == "" or sid == []:
            return "Ошибка: Необходимо выбрать голосовую модель из выпадающего списка 'Voice'", None

        if self.net_g is None or self.pipeline is None:
            logger.warning(f"Модель {sid} не загружена правильно. Попытка повторной загрузки...")
            try:
                self.get_vc(sid, protect)
                if self.net_g is None or self.pipeline is None:
                    logger.error(f"Не удалось загрузить модель {sid} даже после повторной попытки")
                    return "Ошибка: Не удалось загрузить голосовую модель. Попробуйте выбрать другую модель или перезапустить приложение.", None
                logger.info(f"Модель {sid} успешно загружена после повторной попытки")
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели {sid}: {str(e)}")
                return f"Ошибка при загрузке модели: {str(e)}", None
            
        try:
            f0_up_key = int(f0_up_key)
        except ValueError:
            return "Ошибка: Значение высоты тона должно быть числом", None

        input_audio_path = None
        if (isinstance(input_audio_path_uploaded, str) and os.path.isfile(input_audio_path_uploaded)) and (isinstance(input_audio_path_select, str) and os.path.isfile(input_audio_path_select)):
            logger.info("Найдено два источника аудио. Используется загруженный файл.")
            input_audio_path = open(input_audio_path_uploaded, 'rb')
        elif (not (isinstance(input_audio_path_uploaded, str) and os.path.isfile(input_audio_path_uploaded))) and (isinstance(input_audio_path_select, str) and os.path.isfile(input_audio_path_select)):
            logger.info("Используется выбранный файл")
            input_audio_path = open(input_audio_path_select, 'rb')
        else:
            return "Ошибка: Не удалось открыть аудиофайл", None

        try:
            if input_audio_path is None or not hasattr(input_audio_path, 'name'):
                return "Ошибка: Недопустимый аудиофайл", None
                
            logger.info(f"Загрузка аудио из {input_audio_path.name}")
            audio = load_audio(input_audio_path.name, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                logger.info("Загрузка модели Hubert")
                self.hubert_model = load_hubert(self.config)
                if self.hubert_model is None:
                    return "Ошибка: Не удалось загрузить модель Hubert", None

            file_index = (
                (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
                if file_index != ""
                else file_index2
            )

            if self.net_g is None:
                return "Ошибка: Модель голоса не инициализирована", None
            if self.pipeline is None:
                return "Ошибка: Pipeline не инициализирован", None
            if audio is None or len(audio) == 0:
                return "Ошибка: Аудио не загружено или пустое", None

            logger.info("Запуск обработки голоса через pipeline")
            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            
            if audio_opt is None:
                return "Ошибка: Не удалось обработать аудио", None
                
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
                
            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            
            logger.info("Преобразование голоса успешно завершено")
            return (
                "Успешно.\n%s\nВремя:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except Exception as e:
            info = traceback.format_exc()
            logger.error(f"Ошибка при обработке аудио: {str(e)}\n{info}")
            
            if "CUDA out of memory" in info:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "Ошибка: Недостаточно памяти GPU. Попробуйте освободить память или использовать файл меньшего размера.", None
            elif "pipeline" in info and "NoneType" in info:
                return "Ошибка: Необходимо выбрать модель голоса перед конвертацией. Выберите модель из списка и попробуйте снова.", None
            
            return f"Ошибка при обработке аудио: {str(e)}", None

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.basename(path),
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
