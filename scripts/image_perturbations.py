"""
Image-level perturbations for robustness evaluation.
All perturbations are applied at the IMAGE level before feature extraction,
providing ecological validity (simulating real-world image degradation).

5 perturbation types × 3 severity levels = 15 conditions.
"""
import numpy as np
from PIL import Image, ImageFilter
import io
from skimage.color import rgb2hed, hed2rgb

# ============ PERTURBATION DEFINITIONS ============

PERTURBATION_TYPES = {
    'gaussian_noise': {
        'name': 'Gaussian Noise',
        'description': 'Sensor noise simulating camera/scanner variability',
        'severities': [0.02, 0.05, 0.10],  # sigma values
        'severity_labels': ['mild', 'moderate', 'severe'],
    },
    'stain_shift': {
        'name': 'Stain Shift',
        'description': 'HED color shift simulating staining protocol variation',
        'severities': [0.1, 0.2, 0.4],  # magnitude of H/E scaling deviation
        'severity_labels': ['mild', 'moderate', 'severe'],
    },
    'gaussian_blur': {
        'name': 'Gaussian Blur',
        'description': 'Defocus blur simulating scanner focus issues',
        'severities': [0.5, 1.0, 2.0],  # sigma in pixels
        'severity_labels': ['mild', 'moderate', 'severe'],
    },
    'jpeg_compression': {
        'name': 'JPEG Compression',
        'description': 'Compression artifacts simulating storage degradation',
        'severities': [50, 20, 5],  # quality factor (lower = more degradation)
        'severity_labels': ['mild', 'moderate', 'severe'],
    },
    'brightness_contrast': {
        'name': 'Brightness/Contrast Shift',
        'description': 'Calibration drift simulating scanner brightness variation',
        'severities': [0.1, 0.2, 0.4],  # delta brightness / contrast factor
        'severity_labels': ['mild', 'moderate', 'severe'],
    },
}

# ============ PERTURBATION FUNCTIONS ============

def apply_gaussian_noise(img_pil, sigma=0.05):
    """Add Gaussian noise to image."""
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    noise = np.random.randn(*arr.shape) * sigma
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_stain_shift(img_pil, magnitude=0.2):
    """Apply HED stain shift with random direction."""
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    hed = rgb2hed(img_arr)
    # Random direction for stain shift
    h_shift = 1.0 + np.random.choice([-1, 1]) * magnitude
    e_shift = 1.0 - np.random.choice([-1, 1]) * magnitude
    hed[:, :, 0] *= h_shift
    hed[:, :, 1] *= e_shift
    hed = np.clip(hed, 0, 1)
    rgb_aug = hed2rgb(hed)
    rgb_aug = np.clip(rgb_aug * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_aug)

def apply_gaussian_blur(img_pil, sigma=1.0):
    """Apply Gaussian blur."""
    return img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

def apply_jpeg_compression(img_pil, quality=20):
    """Apply JPEG compression artifacts."""
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def apply_brightness_contrast(img_pil, delta=0.2):
    """Apply brightness and contrast shift."""
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    # Random brightness direction
    brightness = np.random.choice([-1, 1]) * delta
    # Contrast reduction
    contrast = 1.0 - delta * 0.5
    arr = (arr - 0.5) * contrast + 0.5 + brightness
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

PERTURBATION_FUNCS = {
    'gaussian_noise': apply_gaussian_noise,
    'stain_shift': apply_stain_shift,
    'gaussian_blur': apply_gaussian_blur,
    'jpeg_compression': apply_jpeg_compression,
    'brightness_contrast': apply_brightness_contrast,
}

# ============ BATCH PERTURBATION ============

def generate_perturbed_images(images_pil, ptype, severity, seed=42):
    """
    Apply a perturbation to a list of PIL images.

    Args:
        images_pil: list of PIL Images
        ptype: perturbation type key
        severity: severity value
        seed: random seed for reproducibility

    Returns:
        list of perturbed PIL Images
    """
    np.random.seed(seed)
    func = PERTURBATION_FUNCS[ptype]
    return [func(img, severity) for img in images_pil]

def get_all_perturbation_conditions():
    """Return list of all (ptype, severity, severity_label) conditions."""
    conditions = []
    for ptype, config in PERTURBATION_TYPES.items():
        for sev, sev_label in zip(config['severities'], config['severity_labels']):
            conditions.append((ptype, sev, sev_label))
    return conditions
