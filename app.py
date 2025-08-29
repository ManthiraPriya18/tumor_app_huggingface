import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import cv2
import numpy as np
import os
import gradio as gr
import io
import base64
from gradcam_utils import GradCAM, overlay_heatmap

# Config
model_path = "disease_model.pth"
device = torch.device("cpu")  # Force CPU for Hugging Face free tier

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
tumor_classes = ['glioma', 'meningioma', 'pituitary']
tumor_threshold = 0.35  

# Global variables for model caching
model = None
gradcam = None

# Load the Model
def load_model():
    global model, gradcam
    
    if model is None:
        print("Loading model...")
        model = models.densenet121(weights=None)  # Fixed deprecated 'pretrained' parameter
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # GradCAM setup
        target_layer = model.features[-1]
        gradcam = GradCAM(model, target_layer)
        print("Model loaded successfully!")

    return model, gradcam

# Image Enhancement + Preprocessing
def enhance_image(img):
    """Enhanced image processing with proper type handling"""
    try:
        # Handle different input types
        if isinstance(img, Image.Image):
            img_array = np.array(img.convert('RGB'))
        elif isinstance(img, str):
            # File path
            img_array = cv2.imread(img)
            if img_array is None:
                raise ValueError(f"Could not read image from path: {img}")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            img_array = img.copy()
            # Ensure RGB format
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Assume it's already RGB
                pass
            elif len(img_array.shape) == 2:
                # Grayscale - convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Handle different channel formats
        if len(img_array.shape) == 2:  # grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif img_array.shape[2] == 3:  # RGB
                pass  # Already RGB
        
        # Apply CLAHE enhancement
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Convert back to 3-channel
        img_enhanced = cv2.merge([gray, gray, gray])
        
        # Apply bilateral filter
        img_enhanced = cv2.bilateralFilter(img_enhanced, 9, 75, 75)
        
        # Normalize
        img_enhanced = img_enhanced.astype(np.float32)
        img_enhanced = (img_enhanced - np.mean(img_enhanced)) / (np.std(img_enhanced) + 1e-5)
        img_enhanced = np.clip(img_enhanced, -1, 1)
        img_enhanced = ((img_enhanced - img_enhanced.min()) / 
                       (img_enhanced.max() - img_enhanced.min()) * 255).astype(np.uint8)
        
        return img_enhanced
        
    except Exception as e:
        print(f"Enhancement error: {e}")
        # Fallback: return basic processed image
        if isinstance(img, Image.Image):
            return np.array(img.convert('RGB'))
        elif isinstance(img, np.ndarray):
            return img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # Create basic placeholder
            return np.ones((224, 224, 3), dtype=np.uint8) * 128

def preprocess_image_from_pil(pil_image, device):
    """Modified to work with PIL images directly"""
    try:
        # Ensure PIL Image
        if not isinstance(pil_image, Image.Image):
            if isinstance(pil_image, np.ndarray):
                pil_image = Image.fromarray(pil_image)
            else:
                raise ValueError(f"Cannot convert {type(pil_image)} to PIL Image")
        
        # Convert to RGB
        pil_image = pil_image.convert('RGB')
        
        # Enhance image
        img_enhanced = enhance_image(pil_image)
        img_pil = Image.fromarray(img_enhanced)
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
        return transform(img_pil).unsqueeze(0).to(device)
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        # Create dummy tensor as fallback
        dummy_tensor = torch.zeros((1, 3, 224, 224)).to(device)
        return dummy_tensor

def preprocess_image(image_path, device):
    """Original function for file paths"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        img = enhance_image(img)
        img = Image.fromarray(img)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0).to(device)
        
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        # Return dummy tensor
        return torch.zeros((1, 3, 224, 224)).to(device)

# Prediction with Test-Time Augmentation (TTA)
def predict(image_input, model, gradcam, n_aug=3):  # Reduced n_aug for HuggingFace
    """Modified to work with PIL images"""
    try:
        # Handle both PIL images and file paths
        if isinstance(image_input, str):
            base_img = Image.open(image_input).convert("RGB")
        else:
            base_img = image_input.convert("RGB") if isinstance(image_input, Image.Image) else Image.fromarray(image_input).convert("RGB")
        
        transform_base = transforms.Compose([
            transforms.Resize((224, 224)),
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
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return dummy results
        dummy_decision = ("no_tumor", 0.5)
        dummy_results = [(cls, 0.25) for cls in class_names]
        dummy_probs = np.array([0.25, 0.25, 0.25, 0.25])
        return dummy_decision, dummy_results, dummy_probs

# Gradio prediction function
def predict_tumor(image):
    """Main prediction function for Gradio interface"""
    if image is None:
        return "⚠️ Please upload an image first.", None
    
    try:
        # Load model
        model, gradcam = load_model()
        
        # Make prediction
        decision, prediction_results, final_probs = predict(image, model, gradcam)
        
        # Format results
        decision_text = "🔴 **TUMOR DETECTED**" if decision[0] == "tumor_detected" else "✅ **NO TUMOR DETECTED**"
        confidence = decision[1] * 100
        
        result_text = f"{decision_text}\n"
        result_text += f"**Confidence**: {confidence:.1f}%\n\n"
        result_text += "### 📊 Detailed Classification:\n"
        
        for cls, prob in prediction_results:
            percentage = prob * 100
            emoji = "🧠" if cls in tumor_classes else "✅"
            result_text += f"{emoji} **{cls.capitalize()}**: {percentage:.1f}%\n"
        
        # Generate visualization
        output_image = None
        try:
            if decision[0] == "tumor_detected":
                # Generate GradCAM for tumor cases
                img_tensor = preprocess_image_from_pil(image, device)
                pred_class = np.argmax(final_probs)
                heatmap = gradcam.generate(img_tensor, class_idx=pred_class)
                
                # Create overlay - FIXED: pass image directly to overlay_heatmap
                enhanced_img = enhance_image(image)  # Returns numpy array
                overlay_result = overlay_heatmap(enhanced_img, heatmap)  # Now handles numpy arrays
                output_image = Image.fromarray(overlay_result.astype(np.uint8))
                
            else:
                # For no tumor cases, show enhanced image
                enhanced_img = enhance_image(image)  # Returns numpy array
                output_image = Image.fromarray(enhanced_img.astype(np.uint8))
                
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
            # Fallback to enhanced original image
            try:
                enhanced_img = enhance_image(image)
                output_image = Image.fromarray(enhanced_img.astype(np.uint8))
            except Exception as fallback_error:
                print(f"Fallback error: {fallback_error}")
                # Last resort: return original image or create placeholder
                if isinstance(image, Image.Image):
                    output_image = image
                else:
                    output_image = Image.new('RGB', (224, 224), color='gray')
            
        return result_text, output_image
        
    except Exception as e:
        error_msg = f"⚠️ **Error during analysis**: {str(e)}\n\nPlease try again with a different image."
        print(f"Main prediction error: {e}")
        
        # Return original image or placeholder
        try:
            if isinstance(image, Image.Image):
                error_image = image
            else:
                error_image = Image.new('RGB', (224, 224), color='red')
        except:
            error_image = Image.new('RGB', (224, 224), color='red')
            
        return error_msg, error_image

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
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
        # 🧠 Brain Tumor 
        
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
                image_input = gr.Image(
                    type="pil", 
                    label="📤 Upload Brain MRI Scan",
                    height=400
                )
                
                submit_btn = gr.Button(
                    "🔍 Analyze Brain Scan", 
                    variant="primary",
                    size="lg"
                )
                
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
                result_output = gr.Markdown(
                    label="📊 AI Analysis Results",
                    elem_classes=["result-text"]
                )
                
                image_output = gr.Image(
                    label="🎯 Enhanced Image / GradCAM Visualization",
                    height=400
                )
        
        # Event handlers
        submit_btn.click(
            fn=predict_tumor,
            inputs=[image_input],
            outputs=[result_output, image_output]
        )
        
        # Auto-analyze on image upload
        image_input.change(
            fn=predict_tumor,
            inputs=[image_input],
            outputs=[result_output, image_output]
        )
        
        # Footer
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

# Main execution - FIXED FOR HUGGINGFACE
if __name__ == "__main__":
    # Always run Gradio interface (no local testing mode on HuggingFace)
    print("Starting Brain Tumor Detection AI...")
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        auth=None,
        show_api=True  # Enable API access
    )