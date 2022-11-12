import time

import gradio as gr
import json
from queue import Queue
import numpy as np
import random
HOME = "/home/yannickfunk/"
css = """
.gradio-container {
    background-image: url("https://i.postimg.cc/4X7vY0xz/background.png");
}
 
h1 {
    color: rgb(60,154,138) !important;
    text-shadow: 1px 0 0 #000, 0 -1px 0 #000, 0 1px 0 #000, -1px 0 0 #000 !important;
    font-size: 80px !important;
}

.gr-button-primary {
    --tw-border-opacity: 1;
    border-color: black;
    --tw-gradient-from: rgb(60,154,138);
    --tw-gradient-to: rgb(108,208,196);
    --tw-gradient-stops: var(--tw-gradient-from), var(--tw-gradient-to);
    --tw-gradient-to: rgb(108,208,196);
    --tw-text-opacity: 1;
    color: white;
}

.gr-button-primary:hover {
    --tw-gradient-to: rgb(60, 154, 138);
}
"""
label_to_index = json.load(open("label_map.json", "r"))
labels = list(label_to_index.keys())
file_to_label = json.load(open("../owlvit/pretrain_labels.json", "r"))
index_to_label = {v: k for k, v in label_to_index.items()}
order = ["01543.jpg", "01714.jpg", "03469.jpg", "14337.jpg",
         "33514.jpg", "20586.jpg", "19832.jpg", "13471.jpg"]


def predict(img, state):
    state_json, count = state
    time.sleep(random.random() * 2)
    file = order[count % len(order)]
    label_vec = np.array(file_to_label[file])
    indices = np.where(label_vec)[0]
    meta_preds = [index_to_label[i] for i in indices]
    pretrained_preds = [index_to_label[i] for i in indices]
    state[0][file] = meta_preds # meta_preds
    state[1] += 1
    return (
        gr.update(value=meta_preds),
        gr.update(value=pretrained_preds),
        state_json,
        state
    )


def update_json(labels, state):
    i = state[1] - 1
    file = order[i]
    state[0][file] = labels
    return state[0], state


title = "Kategorisierer by Team Ananas🍍"


with gr.Blocks(css=css, title=title) as demo:
    state = gr.State([dict(), 0])
    gr.HTML('<div id="bg"></div>')
    gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>"
        + title
        + "</h1>"
    )
    with gr.Row():
        with gr.Column():
            input = gr.Gallery(label="Bildergalerie", value=[HOME+"gradio_files/"+e for e in order])
            predict_btn = gr.Button(value="Kategorisiere!", variant="primary")
        with gr.Column():
            with gr.Row():
                output_meta = gr.CheckboxGroup(label="Kategorien Meta Learner", choices=labels, interactive=True)
                output_pretrained = gr.CheckboxGroup(label="Kategorien Pretrained", choices=labels, interactive=True)
            json_output = gr.JSON(label="Kategorien Meta Learner (JSON)")

    output_meta.change(fn=update_json, inputs=[output_meta, state], outputs=[json_output, state])
    predict_btn.click(fn=predict, inputs=[input, state], outputs=[output_meta, output_pretrained, json_output])

demo.launch()