!pip install datasets scikit-learn gradio numpy

from datasets import load_dataset

dataset = load_dataset("nishant773911/HateXplain", split="train")
# dataset.save_to_disk("HateXplain_local")

dataset
print(dataset[0])

texts = []
labels = []

for example in dataset:
    texts.append(example["text"])
    labels.append(example["label"])

print(texts[:2])
print(labels[:5])

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
model.fit(X, encoded_labels)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, encoded_labels)

print("Training Done ")

def predict_text(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return encoder.inverse_transform([pred])[0]

predict_text("I hate you")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2)

model.fit(X_train, y_train)
preds = model.predict(X_test)

print(classification_report(y_test, preds))

import gradio as gr

def predict_text(text):
    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0]
    pred_idx = prob.argmax()
    pred_label = encoder.inverse_transform([pred_idx])[0]
    return {encoder.classes_[0]: float(prob[0]), encoder.classes_[1]: float(prob[1])}, pred_label

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Title
    gr.Markdown(
        """
        # 🌈 English Text Risk Detector
        **AI-powered text risk detection demo**  
        Enter any English text and see if it contains hate speech.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter Text",
                placeholder="Type your English text here...",
                lines=4
            )
            predict_button = gr.Button("Predict 🚀")
        
        with gr.Column():
            output_probs = gr.Label(label="Probabilities")
            output_label = gr.Textbox(label="Prediction", interactive=False)
    
    # About Us Section
    gr.Markdown(
        """
        ---
        ### About Us
        To showcase AI-based text risk detection.  
        Uses **TF-IDF + Logistic Regression** trained on HateXplain dataset.  

        ### How to Use
        1. Type a sentence in English.  
        2. Press Predict.  
        3. View predicted label and probabilities.
        """
    )
    
    # Connect button
    predict_button.click(
        fn=predict_text,
        inputs=input_text,
        outputs=[output_probs, output_label]
    )

demo.launch()

