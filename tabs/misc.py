import gradio as gr
from tabs.common import i18n

def create_misc_tab():
    with gr.TabItem(i18n("")):
        gr.Markdown('''
            ![arbuz](https://cdn-lfs-us-1.huggingface.co/repos/61/7a/617a4e71eb3a363c7795a9edd65388c15dd1c6e5239ddd68304b96da935b819d/9b724125fabbc1eb055e16033a0571c4da486e5c6f11785afdb6dfbb1f0346ea?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27arbuz.png%3B+filename%3D%22arbuz.png%22%3B&response-content-type=image%2Fpng&Expires=1715533997&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxNTUzMzk5N319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzYxLzdhLzYxN2E0ZTcxZWIzYTM2M2M3Nzk1YTllZGQ2NTM4OGMxNWRkMWM2ZTUyMzlkZGQ2ODMwNGI5NmRhOTM1YjgxOWQvOWI3MjQxMjVmYWJiYzFlYjA1NWUxNjAzM2EwNTcxYzRkYTQ4NmU1YzZmMTE3ODVhZmRiNmRmYmIxZjAzNDZlYT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=hAXARaTl1B7RT0rQCDT7VUQAWB7sKDpk1dr5b5K8Kcs7GbO8WNWZf4hfUasPgLg6~VktTkE6StqgHQO4Aeg5Zv7-seJ9e1rqyrodSHD5Ckjjl~5GRzIRv6uNjHSH1Gt-kupARsfVhq0as0bnpq7K8XBDE7awZLOg~Ikr-9FL6GJQUSD9sMEEr--ar19a1y5v5KC5fe~IIDJB6kfYBBcVQmuARfYtHW6nPPEgiRnhENpjfaZRTjNcO1nQ4Ixw5t4mnm4EINbgMTELVf6laN89mN1BEWU812gKOM45lC4A6sxjhyHo0c-IbwYe5wtklIcJOgun4t4-7cndiLipKwFB5A__&Key-Pair-Id=KCD77M1F0VK2B)
        ''')
        img = gr.Image(
            type="filepath",
            label="Image",
            elem_id="image_component"
        ).style(height=260, width=300)