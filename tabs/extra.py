import gradio as gr
import os
import wave
from tabs.common import (
    i18n, vc, ilariavoices, language_dict, now_dir,
    get_audio_duration, generate_spectrogram_and_get_info, merge, calculate_remaining_time,
    pretrained_G_files, pretrained_D_files, config, names
)
from tabs.inference import (
    tts_and_convert, vc_output1, vc_output2, spk_item, vc_transform0, f0_file,
    f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0,
    rms_mix_rate0, protect0, protect1, file_index4
)

def create_extra_tab():
    with gr.TabItem(i18n("Extra")):
        # Подраздел IlariaTTS
        with gr.Accordion('IlariaTTS', open=False):
            with gr.Column():
                ilariaid = gr.Dropdown(
                    label=i18n("TTS Voice:"), 
                    choices=ilariavoices, 
                    interactive=True, 
                    value="English-Jenny (Female)"
                )
                ilariatext = gr.Textbox(
                    label=i18n("Input your Text"), 
                    interactive=True, 
                    value="Ывывавы, это тест."
                )
                ilariatts_button = gr.Button(value=i18n("Speak and Convert"))
                
                # Для этого обработчика нужны дополнительные переменные, 
                # которые определены в других модулях
                ilariatts_button.click(
                    tts_and_convert,
                    [
                        ilariaid,
                        ilariatext,
                        spk_item,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0
                    ],
                    [vc_output1, vc_output2]
                )
                
        # Подраздел Model Info
        with gr.Accordion(i18n('Model Info'), open=False):
            with gr.Column():
                sid1 = gr.Dropdown(label=i18n("Voice Model"), choices=sorted(names))
                refresh_button = gr.Button(i18n("Refresh"), variant="primary")
                
                modelload_out = gr.Textbox(
                    label=i18n("Model Metadata"), 
                    interactive=False, 
                    lines=4
                )
                get_model_info_button = gr.Button(i18n("Get Model Info"))
                
                get_model_info_button.click(
                    fn=vc.get_vc, 
                    inputs=[sid1, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4, modelload_out]
                )
                
        # Подраздел Audio Analyser
        with gr.Accordion(i18n('Audio Analyser'), open=False):
            with gr.Column():
                audio_input = gr.Audio(type="filepath")
                get_info_button = gr.Button(
                    value=i18n("Get information about the audio"), 
                    variant="primary"
                )
                
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            value=i18n("Information about the audio file"),
                            visible=True,
                        )
                        output_markdown = gr.Markdown(
                            value=i18n("Waiting for information..."), 
                            visible=True
                        )
                        image_output = gr.Image(type="filepath", interactive=False)

            get_info_button.click(
                fn=generate_spectrogram_and_get_info,
                inputs=[audio_input],
                outputs=[output_markdown, image_output],
            )

        # Подраздел Training Helper
        with gr.Accordion(i18n('Training Helper'), open=False):
            with gr.Column():
                audio_input = gr.Audio(
                    type="filepath", 
                    label=i18n("Upload your audio file")
                )
                gr.Text(
                    i18n("Please note that these results are approximate and intended to provide a general idea for beginners."), 
                    label=i18n('Notice:')
                )
                training_info_output = gr.Markdown(label=i18n("Training Information:"))
                get_info_button = gr.Button(i18n("Get Training Info"))
                
                def get_training_info(audio_file):
                    if audio_file is None:
                        return 'Please provide an audio file!'
                    duration = get_audio_duration(audio_file)
                    sample_rate = wave.open(audio_file, 'rb').getframerate()

                    training_info = {
                        (0, 2): (150, 'OV2'),
                        (2, 3): (200, 'OV2'),
                        (3, 5): (250, 'OV2'),
                        (5, 10): (300, 'Normal'),
                        (10, 25): (500, 'Normal'),
                        (25, 45): (700, 'Normal'),
                        (45, 60): (1000, 'Normal')
                    }

                    for (min_duration, max_duration), (epochs, pretrain) in training_info.items():
                        if min_duration <= duration < max_duration:
                            break
                    else:
                        return 'Duration is not within the specified range!'

                    return f'You should use the **{pretrain}** pretrain with **{epochs}** epochs at **{sample_rate/1000}khz** sample rate.'
                
                get_info_button.click(
                    fn=get_training_info,
                    inputs=[audio_input],
                    outputs=[training_info_output]
                )
                
        # Подраздел Training Time Calculator
        with gr.Accordion(i18n('Training Time Calculator'), open=False):
            with gr.Column():
                epochs_input = gr.Number(label=i18n("Number of Epochs"))
                seconds_input = gr.Number(label=i18n("Seconds per Epoch"))
                calculate_button = gr.Button(i18n("Calculate Time Remaining"))
                remaining_time_output = gr.Textbox(
                    label=i18n("Remaining Time"), 
                    interactive=False
                )
                
                calculate_button.click(
                    fn=calculate_remaining_time,
                    inputs=[epochs_input, seconds_input],
                    outputs=[remaining_time_output]
                )
                
        # Подраздел Model Fusion
        with gr.Accordion(i18n("Model Fusion"), open=False):
            with gr.Group():
                gr.Markdown(value=i18n("Strongly suggested to use only very clean models."))
                with gr.Row():
                    ckpt_a = gr.Textbox(
                        label=i18n("Path of the first .pth"), 
                        value="", 
                        interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("Path of the second .pth"), 
                        value="", 
                        interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Weight of the first model over the second"),
                        value=0.5,
                        interactive=True,
                    )
            with gr.Group():
                with gr.Row():
                    sr_ = gr.Radio(
                        label=i18n("Sample rate of both models"),
                        choices=["32k","40k", "48k"],
                        value="32k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=i18n("Pitch Guidance"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=i18n("Add informations to the model"),
                        value="",
                        max_lines=8,
                        interactive=True,
                        visible=False
                    )
                    name_to_save0 = gr.Textbox(
                        label=i18n("Final Model name"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=i18n("Versions of the models"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
            with gr.Group():
                with gr.Row():
                    but6 = gr.Button(i18n("Fuse the two models"), variant="primary")
                    info4 = gr.Textbox(label=i18n("Output"), value="", max_lines=8)
                but6.click(
                    merge,
                    [
                        ckpt_a,
                        ckpt_b,
                        alpha_a,
                        sr_,
                        if_f0_,
                        info__,
                        name_to_save0,
                        version_2,
                    ],
                    info4,
                    api_name="ckpt_merge",
                )

        # Подраздел Credits
        with gr.Accordion(i18n('Credits'), open=False):
            gr.Markdown('''
            ## All the amazing people who worked on this!
            
    ### Developers
    
    - **Ilaria**: Founder, Lead Developer
    - **Yui**: Training feature
    - **GDR-**: Inference feature
    - **Poopmaster**: Model downloader, Model importer
    - **kitlemonfoot**: Ilaria TTS implementation
    - **eddycrack864**: UVR5 implementation
    - **Mikus**: Ilaria Updater & Downloader
    - **Diablo**: Pretrain Automation, UI features, Various fixes
    
    ### Beta Tester
    
    - **Charlotte**: Beta Tester, Advisor
    - **mrm0dz**: Beta Tester, Advisor
    - **RME**: Beta Tester
    - **Delik**: Beta Tester
    - **inductivegrub**: Beta Tester
    - **l3af**: Beta Tester, Helper
    
    ### Pretrains Makers
    
    - **simplcup**: Ov2Super
    - **mustar22**: RIN_E3 & Snowie
    
    ### Colab Port
    
    - **Angetyde**
    - **l3af**
    - **Poopmaster**
    - **Hina**
    
    ### HuggingFace Port
    
    - **Nick088**
    
    ### Other
    
    - **RVC Project**: Original Developers
    - **yumereborn**: Ilaria RVC image
                            
            ### **In loving memory of JLabDX** 🕊️
            ''') 