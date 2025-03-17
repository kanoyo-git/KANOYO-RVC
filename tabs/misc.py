import gradio as gr
from tabs.common import i18n

def create_misc_tab():
    with gr.Tab(label=i18n("Info")):
        # Обновлённый макет для информационной вкладки
        with gr.Blocks():
            gr.Markdown(
                """
                # Kanoyo RVC
                
                ## О проекте
                
                Kanoyo - это инструмент для преобразования голоса на основе RVC.
                
                ## Как использовать
                
                1. Загрузите аудиофайл во вкладке Inference
                2. Выберите модель
                3. Нажмите "Process"
                
                ## Контакты
                
                [GitHub](https://github.com/yourname/kanoyo-rvc)
                """,
                elem_classes=["info-markdown"]
            )
            
            # Обновлённый способ создания ссылки
            gr.HTML(
                """
                <div style="text-align: center; margin-top: 20px;">
                    <a href="https://github.com/yourname/kanoyo-rvc" target="_blank" 
                       style="text-decoration: none; background: #4a4a4a; color: white; 
                              padding: 10px 20px; border-radius: 5px; font-weight: bold;">
                        GitHub
                    </a>
                </div>
                """
            )