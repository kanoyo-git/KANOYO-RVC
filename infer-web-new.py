import gradio as gr
from tabs.common import i18n, css, config
from tabs.inference import create_inference_tab
from tabs.train import create_train_tab
from tabs.extra import create_extra_tab
from tabs.misc import create_misc_tab

# Обновленная инициализация Blocks
app = gr.Blocks(
    css=css,
    theme=gr.themes.Default(),  # Новая поддержка тем
    analytics_enabled=False     # Отключение аналитики, при желании
)

with app:
    gr.HTML("<h1>KANOYO-RVC</h1>")
    
    with gr.Tabs() as tabs:
        with gr.Tab("Inference"):
            inference_tab = tabs.inference.create_inference_tab()
            
        with gr.Tab("Train"):
            train_tab = tabs.train.create_train_tab()
            
        with gr.Tab("Extra"):
            extra_tab = tabs.extra.create_extra_tab()
            
        with gr.Tab("Misc"):
            misc_tab = tabs.misc.create_misc_tab()

# Новый способ настройки очереди
app.queue(
    concurrency_count=3,  # Количество параллельных обработчиков
    max_size=20,          # Максимальный размер очереди
    api_open=False        # Ограничить доступ к API
)

# Обновленный метод запуска
app.launch(
    server_name="0.0.0.0", 
    server_port=7860,
    favicon_path="./assets/favicon.ico",
    share=config.iscolab,
    show_error=True,      # Показывать ошибки в интерфейсе
    quiet=False           # Вывод дополнительной информации в консоль
) 