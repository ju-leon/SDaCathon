import json

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score

num_labels = 37
label_map = {
    "Deckenansicht": 0,
    "Detail": 1,
    "Revisionsoeffnung": 2,
    "Wandansicht": 3,
    "Kabel": 4,
    "Baugeraet": 5,
    "Trockenbau": 6,
    "Tuer": 7,
    "Fertigstellung": 8,
    "Brandschutz": 9,
    "Steckdose_Schalter": 10,
    "Beleuchtung": 11,
    "Lueftung": 12,
    "Fenster": 13,
    "Baumaterial": 14,
    "Aussenbereich": 15,
    "Trasse": 16,
    "Buero": 17,
    "Rohbau": 18,
    "Betonwand": 19,
    "Bauschutt": 20,
    "Abwasser": 21,
    "Heizungsrohr": 22,
    "Durchbruch": 23,
    "Erschliessung": 24,
    "Brandmelder": 25,
    "Keller": 26,
    "Mauerwerk": 27,
    "Sprinkler": 28,
    "Baunotiz": 29,
    "Installation": 30,
    "Bad": 31,
    "Deckenheizung": 32,
    "Technikzentrale": 33,
    "Parkhaus": 34,
    "Schacht": 35,
    "Bodenansicht": 36
}

index_to_label = {v: k for k, v in label_map.items()}

with open("../demo/test.json", "r") as f:
    ground_truth = json.load(f)

with open("testset_preds.json", "r") as f:
    predictions = json.load(f)

files = list(ground_truth.keys())

acc_dict = dict()
recall_dict = dict()
precision_dict = dict()
for i in range(num_labels):
    y_preds = np.array([predictions[file][i] for file in files])
    y_true = np.array([ground_truth[file][i] for file in files])
    acc = accuracy_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds, zero_division=1, average='weighted')
    precision = precision_score(y_true, y_preds, average='weighted', zero_division=1)
    acc_dict[index_to_label[i]] = acc
    recall_dict[index_to_label[i]] = recall
    precision_dict[index_to_label[i]] = precision


with open("acc.json", "w") as f:
    json.dump(acc_dict, f)

with open("recall.json", "w") as f:
    json.dump(recall_dict, f)

with open("precision.json", "w") as f:
    json.dump(precision_dict, f)
