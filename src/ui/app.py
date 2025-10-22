import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

from src.models.resnet50 import ResNet50KL

labels = [f"KL-{i}" for i in range(5)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50KL().to(device)
model.eval()

tf = T.Compose(
    [
        T.Grayscale(num_output_channels=3),
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(img: Image.Image):
    with torch.no_grad():
        x = tf(img).unsqueeze(0).to(device)
        prob = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    return {labels[i]: float(prob[i]) for i in range(5)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Knee X-ray"),
    outputs=gr.Label(num_top_classes=5, label="KL Probability"),
    title="Knee OA Classifier",
    description="무릎 X-ray의 KL 등급 확률을 예측합니다. (가중치 로딩 전 데모)",
)

if __name__ == "__main__":
    demo.launch()
