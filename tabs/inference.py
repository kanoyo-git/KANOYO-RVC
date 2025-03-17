import gradio as gr
from tabs.common import (
    i18n, vc, names, indexes_list, audio_paths, weight_root, index_root, 
    change_choices, clean, css, now_dir, config, F0GPUVisible, import_files,
    language_dict, ilariavoices, asyncio, edge_tts, download_from_url, create_ui_element
)

# Общие переменные для вкладки Inference
vc_output1 = gr.Textbox(label=i18n("Console"))
vc_output2 = gr.Audio(label=i18n("Audio output"))

# Создаём глобальные переменные, используемые в других модулях
spk_item = gr.Slider(
    minimum=0,
    maximum=2333,
    step=1,
    label=i18n("Speaker ID (Auto-Detected)"),
    value=0,
    visible=True,
    interactive=False,
)
vc_transform0 = gr.Slider(
    label=i18n("Pitch: -24 is lower (2 octaves) and 24 is higher (2 octaves)"),
    minimum=-24,
    maximum=24,
    value=0,
    step=1,
)
f0_file = gr.File(
    label=i18n("F0 curve file [optional]"),
    visible=False,
)
f0method0 = gr.Radio(
    label=i18n("Pitch Extraction, rmvpe is best"),
    choices=["harvest", "crepe", "rmvpe"]
    if config.dml is False
    else ["harvest", "rmvpe"],
    value="rmvpe",
    interactive=True,
)
file_index1 = gr.Textbox(
    label=i18n("Path of index"),
    placeholder=".\models\index",
    interactive=True,
    visible=False,
)
file_index2 = gr.Dropdown(
    label=i18n("Auto-detect index path"),
    choices=sorted(indexes_list),
    interactive=True,
)
index_rate1 = gr.Slider(
    minimum=0,
    maximum=1,
    label=i18n("Index Ratio"),
    value=0.40,
    interactive=True,
)
filter_radius0 = gr.Slider(
    minimum=0,
    maximum=7,
    label=i18n(">=3 apply median filter to the harvested pitch results"),
    value=3,
    step=1,
    interactive=True,
)
resample_sr0 = gr.Slider(
    minimum=0,
    maximum=48000,
    label=i18n("Resampling, 0=none"),
    value=0,
    step=1,
    interactive=True,
)
rms_mix_rate0 = gr.Slider(
    minimum=0,
    maximum=1,
    label=i18n("0=Input source volume, 1=Normalized Output"),
    value=0.25,
    interactive=True,
)
protect0 = gr.Slider(
    minimum=0,
    maximum=0.5,
    label=i18n(
        "Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
    value=0.33,
    step=0.01,
    interactive=True,
)
protect1 = gr.Slider(
    minimum=0,
    maximum=0.5,
    label=i18n(
        "Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
    value=0.33,
    step=0.01,
    interactive=True,
)
file_index4 = gr.Dropdown(
    label=i18n("Auto-detect index path"),
    choices=sorted(indexes_list),
    interactive=True,
)

def create_inference_tab():
    with gr.Tab(label=i18n("Inference")):
        with gr.Row():
            sid0 = gr.Dropdown(label=i18n("Voice"), choices=sorted(names))
            sid1 = sid0
            
            with gr.Column():
                refresh_button = gr.Button(i18n("Refresh"), variant="primary")
                clean_button = gr.Button(i18n("Unload Voice from VRAM"), variant="primary")
            vc_transform0.render()
            clean_button.click(
                fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
            )
        
        # Подвкладка Inference
        with gr.TabItem(i18n("Inference")):
            with gr.Group():
                with gr.Row():
                    with gr.Column():                                
                        input_audio = create_ui_element(
                            gr.Audio, 
                            label=i18n("Input audio"), 
                            sources=["upload", "microphone"],
                            type="filepath",
                            elem_classes=["input-audio"]
                        )
                        file_index2.render()
                        input_audio0 = gr.Dropdown(
                            label=i18n("Select a file from the audio folder"),
                            choices=sorted(audio_paths),
                            value='',
                            interactive=True,
                        )
                        refresh_button.click(
                            fn=change_choices,
                            inputs=[],
                            outputs=[sid0, file_index2, input_audio0],
                            api_name="infer_refresh",
                        )
                        file_index1.render()
                    with gr.Column():
                        with gr.Accordion(i18n('Advanced Settings'), open=True, visible=True):
                            with gr.Column():
                                f0method0.render()
                                resample_sr0.render()
                                rms_mix_rate0.render()
                                protect0.render()
                                filter_radius0.render()
                                index_rate1.render()
                                f0_file.render()

                                refresh_button.click(
                                    fn=change_choices,
                                    inputs=[],
                                    outputs=[sid0, file_index2],
                                    api_name="infer_refresh",
                                )
                                spk_item.render()

            with gr.Group():
                with gr.Column():
                    but0 = gr.Button(i18n("Convert"), variant="primary")
                    with gr.Row():
                        vc_output1.render()
                        vc_output2.render()

                    but0.click(
                        vc.vc_single,
                        [
                            spk_item,
                            input_audio0,
                            input_audio,
                            vc_transform0,
                            f0_file,
                            f0method0,
                            file_index1,
                            file_index2,
                            index_rate1,
                            filter_radius0,
                            resample_sr0,
                            rms_mix_rate0,
                            protect0,
                        ],
                        [vc_output1, vc_output2],
                        api_name="infer_convert",
                    )
        
        # Подвкладка Download Voice Models
        with gr.TabItem(i18n("Download Voice Models")):
            gr.Markdown(i18n("For models found in AI Hub"))
            with gr.Row():
                url = gr.Textbox(label=i18n("Huggingface Link:"))
            with gr.Row():
                model = gr.Textbox(label=i18n("Name of the model (without spaces):"))
                download_button = gr.Button(i18n("Download"))
            with gr.Row():
                status_bar = gr.Textbox(label=i18n("Download Status"))
            download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])

        # Подвкладка Import Models
        with gr.TabItem(i18n("Import Models")):
            gr.Markdown(i18n("For models found on Weights"))
            file_upload = gr.File(label=i18n("Upload a .zip file containing a .pth and .index file"))
            import_button = gr.Button(i18n("Import"))
            import_status = gr.Textbox(label=i18n("Import Status"))
            
            def import_button_click(file):
                return import_files(file)
                
            import_button.click(fn=import_button_click, inputs=file_upload, outputs=import_status)

        # Подвкладка Batch Inference
        with gr.TabItem(i18n("Batch Inference")):
            gr.Markdown(
                value=i18n("Batch Conversion")
            )
            
            with gr.Row():
                with gr.Column():
                    vc_transform1 = gr.Number(
                        label=i18n("Pitch: 0 from man to man (or woman to woman); 12 from man to woman and -12 from woman to man."),
                        value=0
                    )
                    opt_input = gr.Textbox(label=i18n("Output"), value="InferOutput")
                    file_index3 = gr.Textbox(
                        label=i18n("Path of index"),
                        placeholder="%userprofile%\\Desktop\\models\\model_example.index",
                        interactive=True,
                    )
                    file_index4.render()
                    f0method1 = gr.Radio(
                        label=i18n("Pitch Extraction, rmvpe is best"),
                        choices=["harvest", "crepe", "rmvpe"]
                        if config.dml is False
                        else ["harvest", "rmvpe"],
                        value="rmvpe",
                        interactive=True,
                    )
                    format1 = gr.Radio(
                        label=i18n("Export Format"),
                        choices=["flac", "wav", "mp3", "m4a"],
                        value="flac",
                        interactive=True,
                    )

                    refresh_button.click(
                        fn=lambda: change_choices()[1],
                        inputs=[],
                        outputs=file_index4,
                        api_name="infer_refresh_batch",
                    )

                with gr.Column():
                    resample_sr1 = gr.Slider(
                        minimum=0,
                        maximum=48000,
                        label=i18n("Resampling, 0=none"),
                        value=0,
                        step=1,
                        interactive=True,
                    )
                    rms_mix_rate1 = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("0=Input source volume, 1=Normalized Output"),
                        value=0.25,
                        interactive=True,
                    )
                    protect1.render()
                    filter_radius1 = gr.Slider(
                        minimum=0,
                        maximum=7,
                        label=i18n(">=3 apply median filter to the harvested pitch results"),
                        value=3,
                        step=1,
                        interactive=True,
                    )
                    index_rate2 = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Index Ratio"),
                        value=0.40,
                        interactive=True,
                    )
            with gr.Row():
                dir_input = gr.Textbox(
                    label=i18n("Enter the path to the audio folder to be processed"),
                    placeholder="%userprofile%\\Desktop\\covers",
                )
                inputs = gr.File(
                    file_count="multiple", label=i18n("Audio files can also be imported in batch")
                )

            with gr.Row():
                but1 = gr.Button(i18n("Convert"), variant="primary")
                vc_output3 = gr.Textbox(label=i18n("Console"))

                but1.click(
                    vc.vc_multi,
                    [
                        spk_item,
                        dir_input,
                        opt_input,
                        inputs,
                        vc_transform1,
                        f0method1,
                        file_index3,
                        file_index4,
                        index_rate2,
                        filter_radius1,
                        resample_sr1,
                        rms_mix_rate1,
                        protect1,
                        format1,
                    ],
                    [vc_output3],
                    api_name="infer_convert_batch",
                )

        # Обработчик для выбора модели голоса
        sid0.change(
            fn=vc.get_vc,
            inputs=[sid0, protect0, protect1],
            outputs=[spk_item, protect0, protect1, file_index2, file_index4, vc_output1],
            api_name="infer_change_voice",
        )

# Функция для TTS и конвертации
def tts_and_convert(ttsvoice, text, spk_item, vc_transform, f0_file, f0method, file_index1, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect):
    # Perform TTS
    vo = language_dict[ttsvoice]
    asyncio.run(edge_tts.Communicate(text, vo).save("./TEMP/temp_ilariatts.mp3"))
    aud_path = './TEMP/temp_ilariatts.mp3'

    # Update output Textbox
    vc_output1.update("Text converted successfully!")

    # Convert voice
    return vc.vc_single(
        spk_item, None, aud_path, vc_transform, f0_file, f0method, 
        file_index1, file_index2, index_rate, filter_radius, 
        resample_sr, rms_mix_rate, protect
    ) 