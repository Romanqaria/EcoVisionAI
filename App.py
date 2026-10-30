
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="wide"
)

MODEL_PATH = Path("models/waste_cnn.pth")
DEVICE = torch.device("cpu")

WASTE_INFO = {
    "cardboard": ("Paper recycling", "Flatten the cardboard and keep it dry."),
    "glass": ("Glass recycling", "Separate glass from general waste. Handle broken glass carefully."),
    "metal": ("Metal recycling", "Separate metal cans and containers for recycling."),
    "paper": ("Paper recycling", "Keep paper clean and dry before recycling."),
    "plastic": ("Plastic recycling", "Check local recycling rules and clean containers where appropriate."),
    "trash": ("General waste", "Check local disposal guidance before throwing this item away.")
}


# -------------------------------
# Model architecture
# Must match train_model.py
# -------------------------------
class WasteCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]

    model = WasteCNN(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return model, classes, preprocess


# -------------------------------
# Prediction
# -------------------------------
def predict_image(image, model, classes, preprocess):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, index = torch.max(probabilities, dim=0)

    return (
        classes[index.item()],
        confidence.item() * 100,
        {
            classes[i]: probabilities[i].item() * 100
            for i in range(len(classes))
        }
    )


# -------------------------------
# User interface
# -------------------------------
st.title("EcoVision AI")
st.subheader("Smart Waste Classification System")
st.write(
    "Upload an image of a waste item and let our trained "
    "Convolutional Neural Network classify it."
)

st.info(
    "This model predicts one of six categories: cardboard, glass, "
    "metal, paper, plastic, or trash. Predictions may be incorrect."
)

try:
    model, classes, preprocess = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

if model is None:
    st.error(
        "Trained model not found. Run `python train_model.py` "
        "from your EcoVisionAI project folder first."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        left, right = st.columns(2)

        with left:
            st.image(
                image,
                caption="Uploaded waste image",
                use_container_width=True
            )

        with right:
            if st.button("Classify Waste", type="primary"):
                with st.spinner("Analyzing image..."):
                    label, confidence, probabilities = predict_image(
                        image, model, classes, preprocess
                    )

                st.success(f"Predicted category: {label.title()}")
                st.metric("Model confidence", f"{confidence:.2f}%")

                if confidence < 50:
                    st.warning(
                        "Confidence is low. Consider trying a clearer image."
                    )

                title, guidance = WASTE_INFO.get(
                    label,
                    ("Disposal guidance", "Check local waste disposal rules.")
                )

                st.markdown(f"### Suggested action: {title}")
                st.write(guidance)

                st.markdown("### Class probabilities")
                st.bar_chart(probabilities)

                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not a guarantee that the prediction is correct."
                )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()
st.caption("EcoVision AI | Artificial Neural Network Lab Project")
