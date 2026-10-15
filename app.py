import gradio as gr
import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download


MODEL_NAME = "microsoft/deberta-v3-base"

MODEL_REPO = "Deepak8409/deberta-v3-emotion-classifier"

MODEL_FILENAME = "best_model.pth"

MAX_LEN = 256

EMOTIONS = [
    "anger",
    "joy",
    "fear",
    "sadness",
    "surprise",
]

# Optimized on validation data
THRESHOLDS = {
    "anger": 0.44,
    "joy": 0.41,
    "fear": 0.59,
    "sadness": 0.59,
    "surprise": 0.47,
}

class EmotionClassifier(nn.Module):

    def __init__(
        self,
        model_name: str,
        num_labels: int = 5,
        dropout: float = 0.3
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            self.transformer.config.hidden_size,
            num_labels
        )

    def forward(self, input_ids, attention_mask):

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = output.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast = False
)

print("Downloading trained model checkpoint...")

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME
)

print(f"Model downloaded to: {model_path}")


# loading model

print("Loading DeBERTa model...")

model = EmotionClassifier(model_name=MODEL_NAME,num_labels=len(EMOTIONS),dropout=0.3)
state_dict = torch.load(model_path,map_location=device,weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("Model loaded successfully!")
# Prediction
def predict_emotions(text):
    if not text or not text.strip():
        return (
            "Please enter some text.",
            {}
        )

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids,attention_mask=attention_mask)
        probabilities = torch.sigmoid(logits)[0]
    probabilities = probabilities.cpu().numpy()
    scores = {
        emotion: float(probabilities[i])
        for i, emotion in enumerate(EMOTIONS)
    }
    predictions = [
        emotion
        for emotion in EMOTIONS
        if scores[emotion] >= THRESHOLDS[emotion]
    ]
    if predictions:
        prediction_text = ", ".join(
            emotion.capitalize()
            for emotion in predictions
        )
    else:
        prediction_text = (
            "No emotion crossed its threshold."
        )
    return prediction_text, scores

# Gradio Interface
description = """
This application uses a fine-tuned **DeBERTa-v3-base** model
for multi-label emotion classification.

### Emotions

Anger • Joy • Fear • Sadness • Surprise

The model uses emotion-specific decision thresholds
optimized on the validation set.

**Validation Macro F1: 0.8460**
"""

demo = gr.Interface(
    fn=predict_emotions,
    inputs=gr.Textbox(label="Enter text",placeholder="Type a sentence...",lines=4),
    outputs=[gr.Textbox(label="Predicted Emotions"),
        gr.Label(label="Emotion Probabilities",num_top_classes=5),],
    title="DeBERTa Multi-Label Emotion Classifier",
    description=description,
)
# Launch
if __name__ == "__main__":

    demo.launch()