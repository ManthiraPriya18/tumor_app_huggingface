import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        b, k, u, v = gradients.size()

        # global-average-pool gradients
        alpha = gradients.view(b, k, -1).mean(2)

        weights = alpha.view(b, k, 1, 1)
        cam = (weights * activations).sum(1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        return cam

def overlay_heatmap(img_input, heatmap, out_path=None, alpha=0.4):
    """
    Overlay heatmap on original image
    
    Args:
        img_input: Can be PIL Image, numpy array, or file path
        heatmap: Numpy array of heatmap values
        out_path: Optional path to save result
        alpha: Overlay transparency (0.0-1.0)
    
    Returns:
        Numpy array of overlaid image (RGB format)
    """
    try:
        # Handle different input types
        if isinstance(img_input, str):
            # File path
            orig = cv2.imread(img_input)
            if orig is None:
                raise ValueError(f"Could not read image from path: {img_input}")
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, Image.Image):
            # PIL Image
            orig = np.array(img_input.convert('RGB'))
        elif isinstance(img_input, np.ndarray):
            # Numpy array
            orig = img_input.copy()
            # Convert BGR to RGB if needed
            if orig.shape[2] == 3 and orig.dtype == np.uint8:
                # Check if it might be BGR (common with OpenCV)
                orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")
        
        # Ensure proper size
        if orig.shape[:2] != (224, 224):
            orig = cv2.resize(orig, (224, 224))
        
        # Ensure heatmap is proper format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        
        # Normalize heatmap to 0-255
        heatmap_normalized = np.uint8(255 * heatmap)
        
        # Apply colormap (JET colormap for visualization)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Convert heatmap from BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure both images are same size
        if heatmap_colored.shape[:2] != orig.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (orig.shape[1], orig.shape[0]))
        
        # Create overlay
        overlay = cv2.addWeighted(
            orig.astype(np.uint8), 
            1 - alpha, 
            heatmap_colored.astype(np.uint8), 
            alpha, 
            0
        )
        
        # Save if output path provided
        if out_path:
            save_img = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, save_img)
        
        return overlay
        
    except Exception as e:
        print(f"Error in overlay_heatmap: {e}")
        # Return original image as fallback
        if isinstance(img_input, Image.Image):
            return np.array(img_input.convert('RGB'))
        elif isinstance(img_input, np.ndarray):
            return img_input
        else:
            # Create placeholder image
            return np.zeros((224, 224, 3), dtype=np.uint8)