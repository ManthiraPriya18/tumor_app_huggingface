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
        
        # Use recommended full backward hook
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
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
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        b, k, u, v = gradients.size()

        # global-average-pool gradients to get weights
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        cam = (weights * activations).sum(1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        # Normalize cam to 0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # Do NOT resize here, handle resizing externally to match original image size
        return cam

def overlay_heatmap(img_input, heatmap, out_path=None, alpha=0.4):
    """
    Overlay heatmap on original image
    
    Args:
        img_input: PIL Image, numpy array, or file path
        heatmap: numpy array heatmap normalized 0-1
        out_path: optional path to save overlaid image
        alpha: blend factor 0-1 for heatmap transparency
    
    Returns:
        numpy array (RGB) with heatmap overlay
    """
    try:
        # Load or convert original image to numpy RGB
        if isinstance(img_input, str):
            orig = cv2.imread(img_input)
            if orig is None:
                raise ValueError(f"Cannot read image from path: {img_input}")
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, Image.Image):
            orig = np.array(img_input.convert('RGB'))
        elif isinstance(img_input, np.ndarray):
            orig = img_input.copy()
            if orig.shape[2] == 3 and orig.dtype == np.uint8:
                orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        heatmap_normalized = np.uint8(255 * heatmap_resized)

        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(orig.astype(np.uint8), 1 - alpha, heatmap_colored.astype(np.uint8), alpha, 0)

        if out_path:
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return overlay
    except Exception as e:
        print(f"Error in overlay_heatmap: {e}")
        if isinstance(img_input, Image.Image):
            return np.array(img_input.convert('RGB'))
        elif isinstance(img_input, np.ndarray):
            return img_input
        else:
            return np.zeros((224, 224, 3), dtype=np.uint8)
