import gradio as gr
import os
import pathlib
import json
import numpy as np
from random import shuffle
from subprocess import Popen
import threading
from sklearn.cluster import MiniBatchKMeans
import faiss
import logging
import traceback

from tabs.common import (
    i18n, now_dir, config, get_pretrained_models, sr_dict, F0GPUVisible,
    gpu_info, default_batch_size, gpus, if_done, if_done_multi, pretrained_G_files, 
    pretrained_D_files, logger
)

def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)

def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["32k","40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["32k", "40k", "48k", "OV2-32k", "OV2-40k", "RIN-40k","Snowie-40k","Snowie-48k","Italia-32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )

def change_f0(if_f0_3, sr2, version19):
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 is True else "", sr2),
    )

def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield f.read()
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                    '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                    % (
                        config.python_cmd,
                        now_dir,
                        exp_dir,
                        n_p,
                        f0method,
                    )
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                            '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                            % (
                                config.python_cmd,
                                leng,
                                idx,
                                n_g,
                                now_dir,
                                exp_dir,
                                config.is_half,
                            )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )
                    ps.append(p)
                done = [False]
                threading.Thread(
                    target=if_done_multi,
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                        config.python_cmd
                        + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                        % (
                            now_dir,
                            exp_dir,
                        )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )
                p.wait()
                done = [True]
        while 1:
            with open(
                    "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield f.read()
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log

    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
                '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
                % (
                    config.python_cmd,
                    config.device,
                    leng,
                    idx,
                    n_g,
                    now_dir,
                    exp_dir,
                    version19,
                )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )
        ps.append(p)
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield f.read()
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log

def train_index(exp_dir1, version19):
    exp_dir = "logs/%s" % exp_dir1
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    if not os.path.exists(feature_dir):
        return "Please perform Feature Extraction First!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform Feature Extraction First！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "Success，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    yield "\n".join(infos)

def click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % exp_dir
        f0nsf_dir = "%s/2b-f0nsf" % exp_dir
        names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy"
                "|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s '
                "-sw %s -v %s"
                % (
                    config.python_cmd,
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
        )
    else:
        cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw '
                "%s -v %s"
                % (
                    config.python_cmd,
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
        )
    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "You can view console or train.log"

def transfer_files(filething, dataset_dir='dataset/'):
    file_names = [f.name for f in filething]
    for f in file_names:
        filename = os.path.basename(f)
        destination = os.path.join(dataset_dir, filename)
        shutil.copyfile(f, destination)
    return i18n("Transferred files to dataset directory!")

def create_train_tab():
    with gr.TabItem(i18n("Train")):
        gr.Markdown(value=i18n(""))
        with gr.Row():
            exp_dir1 = gr.Textbox(label=i18n("Model Name"), value="test-model")
            sr2 = gr.Dropdown(
                label=i18n("Sample Rate & Pretrain"),
                choices=["OV2-40k","SnowieV3.1-40k","SnowieV3.1-48k"],
                value="ВЫБЕРИТЕ ПРЕТРЕЙН",
                interactive=True,
            )
            version19 = gr.Radio(
                label=i18n("Version 2 only here"),
                choices=["v2"],
                value="v2",
                interactive=False,
                visible=False,
            )
            np7 = gr.Slider(
                minimum=0,
                maximum=config.n_cpu,
                step=1,
                label=i18n("CPU Threads"),
                value=int(np.ceil(config.n_cpu / 2.5)),
                interactive=True,
            )
        with gr.Group():
            gr.Markdown(value=i18n(""))
            with gr.Row():
                trainset_dir4 = gr.Textbox(
                    label=i18n("Path to Dataset"), value="dataset"
                )
                with gr.Accordion(i18n('Upload Dataset (alternative)'), open=False, visible=True):
                    file_thin = gr.Files(label=i18n('Audio Files'))
                    show = gr.Textbox(label=i18n('Status'))
                    transfer_button = gr.Button(i18n('Upload Dataset to the folder'), variant="primary")
                    transfer_button.click(
                        fn=transfer_files,
                        inputs=[file_thin],
                        outputs=show,
                    )

        with gr.Group():
            gr.Markdown(value=i18n(""))
            with gr.Row():
                save_epoch10 = gr.Slider(
                    minimum=1,
                    maximum=250,
                    step=1,
                    label=i18n("Save frequency"),
                    value=50,
                    interactive=True,
                )
                total_epoch11 = gr.Slider(
                    minimum=2,
                    maximum=10000,
                    step=1,
                    label=i18n("Total Epochs"),
                    value=300,
                    interactive=True,
                )
                batch_size12 = gr.Slider(
                    minimum=1,
                    maximum=16,
                    step=1,
                    label=i18n("Batch Size"),
                    value=default_batch_size,
                    interactive=True,
                )
                if_save_every_weights18 = gr.Radio(
                    label=i18n("Create model with save frequency"),
                    choices=[i18n("是"), i18n("否")],
                    value=i18n("是"),
                    interactive=True,
                )

        with gr.Accordion(i18n('Advanced Settings'), open=False, visible=True):
            with gr.Row(): 
                with gr.Group():
                    spk_id5 = gr.Slider(
                            minimum=0,
                            maximum=4,
                            step=1,
                            label=i18n("Speaker ID"),
                            value=0,
                            interactive=True,
                        )
                    if_f0_3 = gr.Radio(
                    label=i18n("Pitch Guidance"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                    gpus6 = gr.Textbox(
                            label=i18n("GPU ID (Leave 0 if you have only one GPU, use 0-1 for multiple GPus)"),
                            value=gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                    gpu_info9 = gr.Textbox(
                            label=i18n("GPU Model"),
                            value=gpu_info,
                            visible=F0GPUVisible,
                        )
                    gpus16 = gr.Textbox(
                    label=i18n("Enter cards to be used (Leave 0 if you have only one GPU, use 0-1 for multiple GPus)"),
                    value=gpus if gpus != "" else "0",
                    interactive=True,
                    )
                    with gr.Group():
                        if_save_latest13 = gr.Radio(
                            label=i18n("Save last ckpt as final Model"),
                            choices=[i18n("是"), i18n("否")],
                            value=i18n("是"),
                            interactive=True,
                        )
                        if_cache_gpu17 = gr.Radio(
                            label=i18n("Cache data to GPU (Only for datasets under 8 minutes)"),
                            choices=[i18n("是"), i18n("否")],
                            value=i18n("否"),
                            interactive=True,
                        )
                        f0method8 = gr.Radio(
                                label=i18n("Feature Extraction Method"),
                                choices=["rmvpe", "rmvpe_gpu"],
                                value="rmvpe_gpu",
                                interactive=True,
                            )
                        gpus_rmvpe = gr.Textbox(
                                label=i18n(
                                    "rmvpe_gpu will use your GPU instead of the CPU for the feature extraction"
                                ),
                                value="%s-%s" % (gpus, gpus),
                                interactive=True,
                                visible=F0GPUVisible,
                            )
                        f0method8.change(
                            fn=change_f0_method,
                            inputs=[f0method8],
                            outputs=[gpus_rmvpe],
                        )        

        with gr.Row():
            pretrained_G14 = gr.Dropdown(
                label="Pretrained G",
                choices=list(pretrained_G_files.values()),
                value=pretrained_G_files.get('f0G32.pth', ''),
                visible=False,
                interactive=True,
            )
            pretrained_D15 = gr.Dropdown(
                label="Pretrained D",
                choices=list(pretrained_D_files.values()),
                value=pretrained_D_files.get('f0D32.pth', ''),
                visible=False,
                interactive=True,
            )
            sr2.change(
                change_sr2,
                [sr2, if_f0_3, version19],
                [pretrained_G14, pretrained_D15],
            )
            version19.change(
                change_version19,
                [sr2, if_f0_3, version19],
                [pretrained_G14, pretrained_D15, sr2],
            )
            if_f0_3.change(
                change_f0,
                [if_f0_3, sr2, version19],
                [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
            )
        
        with gr.Group():
            with gr.Row():
                but1 = gr.Button(i18n("1. Process Data"), variant="primary")
                but2 = gr.Button(i18n("2. Feature Extraction"), variant="primary")
                but4 = gr.Button(i18n("3. Train Index"), variant="primary")
                but3 = gr.Button(i18n("4. Train Model"), variant="primary")
                info = gr.Textbox(label=i18n("Output"), value="", max_lines=5, lines=5)
                but1.click(
                   preprocess_dataset,
                       [trainset_dir4, exp_dir1, sr2, np7],
                       [info],
                       api_name="train_preprocess",
                    )
                but2.click(
                   extract_f0_feature,
                       [
                           gpus6,
                           np7,
                           f0method8,
                           if_f0_3,
                           exp_dir1,
                           version19,
                           gpus_rmvpe,
                       ],
                       [info],
                       api_name="train_extract_f0_feature",
                )
                but4.click(train_index, [exp_dir1, version19], info)
                but3.click(
                   click_train,
                   [
                       exp_dir1,
                       sr2,
                       if_f0_3,
                       spk_id5,
                       save_epoch10,
                       total_epoch11,
                       batch_size12,
                       if_save_latest13,
                       pretrained_G14,
                       pretrained_D15,
                       gpus16,
                       if_cache_gpu17,
                       if_save_every_weights18,
                       version19,
                   ],
                   info,
                   api_name="train_start",
                )
                but4.click(train_index, [exp_dir1, version19], info)

        gr.Examples(
            examples=[
                ["path/to/audio1.wav", 32000],
                ["path/to/audio2.wav", 40000]
            ],
            inputs=[audio_input, sr_select],
            label="Training Examples"
        )

    return train_block 