import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import cv2
import numpy as np
import os
import gradio as gr
from gradcam_utils import GradCAM, overlay_heatmap

# Config
model_path = "disease_model.pth"
device = torch.device("cpu")  # Force CPU for Hugging Face free tier

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
tumor_classes = ['glioma', 'meningioma', 'pituitary']
tumor_threshold = 0.35  

model = None
gradcam = None

def load_model():
    global model, gradcam
    if model is None:
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        target_layer = model.features.denseblock4.denselayer16.conv2
        gradcam = GradCAM(model, target_layer)
    return model, gradcam

def enhance_image(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    img = cv2.merge([gray, gray, gray])
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = img.astype(np.float32)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    img = np.clip(img, -1, 1)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return img

def preprocess_image_from_pil(pil_image, device):
    pil_image = pil_image.convert('RGB')
    img_enhanced = enhance_image(np.array(pil_image))
    img_pil = Image.fromarray(img_enhanced)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img_pil).unsqueeze(0).to(device)

def predict(image_input, model, gradcam, n_aug=3):
    if isinstance(image_input, str):
        base_img = Image.open(image_input).convert("RGB")
    else:
        base_img = image_input.convert("RGB") if isinstance(image_input, Image.Image) else Image.fromarray(image_input).convert("RGB")

    transform_base = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    probs_list = []
    with torch.no_grad():
        for _ in range(n_aug):
            aug = transforms.RandomApply([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ], p=0.7)
            aug_img = aug(base_img)
            img_tensor = transform_base(aug_img).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
    final_probs = np.mean(probs_list, axis=0).flatten()
    tumor_prob = sum([final_probs[class_names.index(cls)] for cls in tumor_classes])
    no_tumor_prob = final_probs[class_names.index("notumor")]

    if tumor_prob > tumor_threshold:
        decision = ("tumor_detected", float(tumor_prob))
    else:
        decision = ("no_tumor", float(no_tumor_prob))

    results = [(class_names[i], float(final_probs[i])) for i in range(len(class_names))]
    results.sort(key=lambda x: x[1], reverse=True)
    return decision, results, final_probs

def predict_tumor(image):
    if image is None:
        return "⚠️ Please upload an image first.", None
    try:
        model, gradcam = load_model()
        decision, prediction_results, final_probs = predict(image, model, gradcam)

        decision_text = "🔴 **TUMOR DETECTED**" if decision[0] == "tumor_detected" else "✅ **NO TUMOR DETECTED**"
        confidence = decision[1] * 100
        result_text = f"{decision_text}\n**Confidence**: {confidence:.1f}%\n\n### 📊 Detailed Classification:\n"
        for cls, prob in prediction_results:
            emoji = "🧠" if cls in tumor_classes else "✅"
            result_text += f"{emoji} **{cls.capitalize()}**: {prob*100:.1f}%\n"

        output_image = None
        try:
            if decision[0] == "tumor_detected":
                img_tensor = preprocess_image_from_pil(image, device)
                pred_class = np.argmax(final_probs)
                heatmap = gradcam.generate(img_tensor, class_idx=pred_class)
                enhanced_img = enhance_image(np.array(image))
                overlay_result = overlay_heatmap(enhanced_img, heatmap)
                output_image = Image.fromarray(overlay_result.astype(np.uint8))
            else:
                enhanced_img = enhance_image(np.array(image))
                output_image = Image.fromarray(enhanced_img.astype(np.uint8))
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
            output_image = image if isinstance(image, Image.Image) else Image.new('RGB', (224,224), color='gray')

        return result_text, output_image
    except Exception as e:
        error_msg = f"⚠️ **Error during analysis**: {str(e)}\n\nPlease try again with a different image."
        print(f"Main prediction error: {e}")
        error_image = image if isinstance(image, Image.Image) else Image.new('RGB', (224,224), color='red')
        return error_msg, error_image

def create_interface():
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .result-text {
        font-size: 16px;
        line-height: 1.6;
    }
    """
    with gr.Blocks(title="Brain Tumor Detection AI", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 Brain Tumor Detection

        **Advanced AI-powered brain tumor detection using DenseNet-121**

        Upload an MRI brain scan to detect and classify potential tumors:
        - **Glioma** - Malignant brain tumor
        - **Meningioma** - Usually benign tumor  
        - **Pituitary** - Pituitary gland tumor
        - **No Tumor** - Healthy brain scan

        ⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. 
        Always consult qualified medical professionals for diagnosis and treatment.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="📤 Upload Brain MRI Scan", height=400)
                submit_btn = gr.Button("🔍 Analyze Brain Scan", variant="primary", size="lg")
                gr.Markdown("""
                ### 📋 How to use:
                1. **Upload** a brain MRI scan (JPEG, PNG)
                2. **Click** "Analyze Brain Scan" 
                3. **View** AI prediction results
                4. **See** GradCAM visualization for tumors

                ### 🔬 AI Features:
                - **Test-Time Augmentation** for better accuracy
                - **Image Enhancement** with CLAHE
                - **GradCAM Visualization** for tumor localization
                - **Multi-class Classification** with confidence scores
                """)
            with gr.Column(scale=1):
                result_output = gr.Markdown(label="📊 AI Analysis Results", elem_classes=["result-text"])
                image_output = gr.Image(label="🎯 Enhanced Image / GradCAM Visualization", height=400)
        submit_btn.click(fn=predict_tumor, inputs=[image_input], outputs=[result_output, image_output])
        image_input.change(fn=predict_tumor, inputs=[image_input], outputs=[result_output, image_output])
        gr.Markdown("""
        ---
        ### 🔬 Technical Details:
        - **Model**: DenseNet-121 architecture
        - **Training**: Medical brain MRI dataset
        - **Techniques**: Test-Time Augmentation, CLAHE enhancement
        - **Visualization**: GradCAM attention maps
        - **API**: REST endpoint available (see API tab above)

        ### 📊 Model Performance:
        - Trained on thousands of brain MRI scans
        - Multi-class tumor classification
        - Enhanced preprocessing pipeline
        - Confidence-based predictions

        **🔧 Built with**: PyTorch • OpenCV • Gradio • HuggingFace Spaces
        """)

    return demo

if __name__ == "__main__":
    print("Starting Brain Tumor Detection AI...")
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, auth=None, show_api=True)
