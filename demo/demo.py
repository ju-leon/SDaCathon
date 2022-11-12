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
print(file_to_label.keys())
order = [
    "01543.jpg", "01714.jpg", "03469.jpg", "14337.jpg",
    "33514.jpg", "20586.jpg", "19832.jpg", "13471.jpg",
    '10220.jpg', '09187.jpg', '19926.jpg', '32843.jpg',
    '23213.jpg', '28527.jpg', '01820.jpg', '16067.jpg',
    '15319.jpg', '13390.jpg', '19934.jpg', '23248.jpg',
    '22213.jpg', '05181.jpg', '12439.jpg', '29412.jpg',
    '21969.jpg', '03134.jpg', '18664.jpg', '30238.jpg',
    '03550.jpg', '01830.jpg', '29962.jpg', '11710.jpg',
    '13393.jpg', '15845.jpg', '08589.jpg', '06191.jpg',
    '22269.jpg', '28219.jpg', '19349.jpg', '08986.jpg',
    '18967.jpg', '26263.jpg', '29452.jpg', '03062.jpg',
    '06310.jpg', '24878.jpg', '24392.jpg', '19242.jpg',
    '22175.jpg', '25149.jpg', '18146.jpg', '08181.jpg',
    '11960.jpg', '04112.jpg', '20736.jpg', '15250.jpg',
    '12440.jpg', '05055.jpg', '15821.jpg', '22901.jpg',
    '18635.jpg', '12504.jpg', '25939.jpg', '21385.jpg',
    '07993.jpg', '11987.jpg', '05342.jpg'
]


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
    file = order[i % len(order)]
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
            input = gr.Gallery(label="Bildergalerie", value=[HOME+"bildarchivierung/data/pretrain/images/"+e for e in order])
            predict_btn = gr.Button(value="Kategorisiere!", variant="primary")
        with gr.Column():
            with gr.Row():
                output_meta = gr.CheckboxGroup(label="Kategorien MAML", choices=labels, interactive=True)
                output_pretrained = gr.CheckboxGroup(label="Kategorien OWL-ViT", choices=labels, interactive=True)
            json_output = gr.JSON(label="Kategorien MAML (JSON)")

    output_meta.change(fn=update_json, inputs=[output_meta, state], outputs=[json_output, state])
    predict_btn.click(fn=predict, inputs=[input, state], outputs=[output_meta, output_pretrained, json_output])

demo.launch()