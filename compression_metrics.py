# Import libraries
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim



# Load images from paths
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return image

# Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(original_path, compressed_path):
    original = load_image(original_path)
    compressed = load_image(compressed_path)
    try:
        mse = np.mean((original - compressed) ** 2)
    except Exception:
        target_height, target_width = compressed.shape[:2]
        original = cv2.resize(original, (target_width, target_height), interpolation=cv2.INTER_AREA)
        mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # No difference
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original_path, compressed_path):
    original = load_image(original_path)
    compressed = load_image(compressed_path)
    
    # Set win_size to be the minimum of image dimensions (height, width) or 7, and set channel_axis for color images
    min_dim = min(original.shape[:2])  # Get the smaller dimension of the image
    win_size = min(min_dim, 7)  # Ensure win_size is not larger than the smallest dimension
    try:
        ssim_index, _ = ssim(original, compressed, full=True, multichannel=True, win_size=win_size, channel_axis=-1)
    except Exception:
        target_height, target_width = compressed.shape[:2]
        original = cv2.resize(original, (target_width, target_height), interpolation=cv2.INTER_AREA)
        ssim_index, _ = ssim(original, compressed, full=True, multichannel=True, win_size=win_size, channel_axis=-1)
    return ssim_index

# Simulate Multiscale SSIM (MS-SSIM)
def calculate_ms_ssim(original_path, compressed_path):
    def calculate_ssim_from_array(original, compressed):
        # Set win_size to be the minimum of image dimensions (height, width) or 7, and set channel_axis for color images
        min_dim = min(original.shape[:2])  # Get the smaller dimension of the image
        win_size = min(min_dim, 7)  # Ensure win_size is not larger than the smallest dimension
        ssim_index, _ = ssim(original, compressed, full=True, multichannel=True, win_size=win_size, channel_axis=-1)
        return ssim_index
    original = load_image(original_path)
    compressed = load_image(compressed_path)
    if original.shape != compressed.shape:
        target_height, target_width = compressed.shape[:2]
        original = cv2.resize(original, (target_width, target_height), interpolation=cv2.INTER_AREA)
    # Downscale images and compute SSIM at different scales
    scales = [1, 0.5, 0.25]  # Different scales (original, half, quarter resolution)
    ms_ssim_value = 0
    for scale in scales:
        scaled_original = cv2.resize(original, (0, 0), fx=scale, fy=scale)
        scaled_compressed = cv2.resize(compressed, (0, 0), fx=scale, fy=scale)
        ms_ssim_value += calculate_ssim_from_array(scaled_original, scaled_compressed)
    return ms_ssim_value / len(scales)

# Bitrate (Bits Per Pixel)
def calculate_bpp(original_path, compressed_file_path):
    original = load_image(original_path)
    
    height, width = original.shape[:2]
    num_pixels = height * width
    compressed_size = os.path.getsize(compressed_file_path) * 8  # Convert bytes to bits
    bpp = compressed_size / num_pixels
    return bpp

# Compression Ratio
def calculate_compression_ratio(original_file_path, compressed_file_path):
    original_size = os.path.getsize(original_file_path)
    compressed_size = os.path.getsize(compressed_file_path)
    return original_size / compressed_size
