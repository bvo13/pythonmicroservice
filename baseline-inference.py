import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
import torch
import torchaudio
from voice_emotion_classification import EmotionClassificationPipeline





root_dir = "C:\\Users\\brady\\cremadset"

classifier = EmotionClassificationPipeline.from_pretrained("griko/emotion_8_cls_svm_ecapa_ravdess")


def get_emotion(file):
    filename_attributes = file.split("_")
    emotion = filename_attributes[2]
    return emotion

audios = []
true_labels =[]
predicted_labels=[]
label_mapping = {
    "calm": "NEU",
    "neutral": "NEU",
    "happy": "HAP",
    "sad": "SAD",
    "angry": "ANG",
    "fearful": "FEA",
    "disgust": "DIS",
    "surprised": "HAP"
}

for actor in os.listdir(root_dir):
    actor_path = os.path.join(root_dir,actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(root_dir, actor, file)
                label = get_emotion(file_path)
                audios.append(file_path)
                true_labels.append(label)

print(f"Loaded {len(audios)} audio files with labels.")



train_files, test_files, train_labels, test_labels = train_test_split(
    audios, true_labels, test_size=0.2, random_state=42, stratify=true_labels
)
for audio in test_files:
    result = classifier(audio)
    predicted_label = result[0]
    predicted_labels.append(label_mapping.get(predicted_label,"NEU"))

f1 = f1_score(test_labels, predicted_labels, average='weighted')
print(f"F1 score is: {f1}")