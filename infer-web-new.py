import gradio as gr
from tabs.common import i18n, css, config
from tabs.inference import create_inference_tab
from tabs.train import create_train_tab
from tabs.extra import create_extra_tab
from tabs.misc import create_misc_tab

# Создаем интерфейс приложения
with gr.Blocks(theme=gr.themes.Base(), title="Kanoyo", css=css) as app:
    gr.HTML('''
        <h1 style="display: flex; align-items: center;">
                <img src="https://art.pixilart.com/sr220411b1340ff.png" alt="heart" style="width:42px;height:42px;border-radius:10%;margin-right:10px;">
                Kanoyo
        </h1>
    ''')
    gr.HTML(
        "<h3>Самая базированная база 👻</h3>"
    )
    
    # Создаем вкладки при помощи модулей
    with gr.Tabs():
        create_inference_tab()
        create_train_tab()  
        create_extra_tab()
        create_misc_tab()

# Запускаем приложение
if config.iscolab:
    app.launch(
        share=True,
        port=config.listen_port,
        favicon=("./assets/favicon.ico"),
    )
else:
    app.launch(
        host="0.0.0.0",
        show_error=True,
        show_api=False,
        open_browser=not config.noautoopen,
        port=config.listen_port,
        favicon=("./assets/favicon.ico"),
        quiet=True,
    ) 