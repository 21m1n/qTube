import gradio as gr

def greet(url, api):
    return "Hello, this is the url: " + url + " api : " + api + "!"

demo = gr.Interface(
    fn=greet,
    inputs=["text", "text"],
    outputs=[gr.Textbox(label="Output", lines=4)],
    allow_flagging=False,
)

demo.launch()
