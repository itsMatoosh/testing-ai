"""
Perturbation Analysis Script for Adversarial Attack Results

This script analyzes the perturbations introduced by different adversarial attacks
(FGM, PGD, HC) on images, comparing them against clean versions.

Metrics computed:
- L0 (pixels changed): Count of pixels where any channel differs significantly
- L-infinity: Maximum absolute perturbation across all pixels/channels
- L2: Euclidean norm of perturbation (normalized)
- Average perturbation per pixel
- SSIM: Structural Similarity Index
- Per-channel statistics (R, G, B)
- Perturbation coverage percentage
"""

import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


# ============================================================
# SSIM Implementation (simplified, no skimage dependency)
# ============================================================

def compute_ssim(img1: np.ndarray, img2: np.ndarray, win_size: int = 7) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image as numpy array (H, W, C) with values in [0, 1]
        img2: Second image as numpy array (H, W, C) with values in [0, 1]
        win_size: Window size for local statistics
        
    Returns:
        SSIM value between -1 and 1 (1 = identical)
    """
    C1 = 0.01 ** 2  # Stability constant for luminance
    C2 = 0.03 ** 2  # Stability constant for contrast
    
    # Convert to grayscale for SSIM computation
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
        img2 = np.mean(img2, axis=2)
    
    # Local means
    mu1 = uniform_filter(img1, size=win_size, mode='reflect')
    mu2 = uniform_filter(img2, size=win_size, mode='reflect')
    
    # Local variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(img1 ** 2, size=win_size, mode='reflect') - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=win_size, mode='reflect') - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=win_size, mode='reflect') - mu1_mu2
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


# ============================================================
# Metric Computation Functions
# ============================================================

def compute_metrics(clean_img: np.ndarray, attacked_img: np.ndarray, 
                   pixel_threshold: float = 1/255) -> Dict:
    """
    Compute all perturbation metrics between clean and attacked images.
    
    Args:
        clean_img: Clean image as numpy array (H, W, C) with values in [0, 1]
        attacked_img: Attacked image as numpy array (H, W, C) with values in [0, 1]
        pixel_threshold: Threshold for considering a pixel as "changed"
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Compute perturbation
    perturbation = attacked_img - clean_img
    abs_perturbation = np.abs(perturbation)
    
    H, W, C = clean_img.shape
    total_pixels = H * W
    
    # L-infinity (maximum perturbation)
    l_inf = float(np.max(abs_perturbation))
    
    # L2 norm (normalized by number of elements)
    l2 = float(np.sqrt(np.sum(perturbation ** 2)))
    l2_normalized = l2 / np.sqrt(H * W * C)
    
    # L0 - count pixels where any channel changed significantly
    max_channel_diff = np.max(abs_perturbation, axis=2)  # Max diff per pixel
    pixels_changed = int(np.sum(max_channel_diff > pixel_threshold))
    pixels_changed_pct = 100 * pixels_changed / total_pixels
    
    # Average perturbation per pixel (across all channels)
    avg_perturbation = float(np.mean(abs_perturbation))
    
    # Per-channel statistics
    channel_names = ['R', 'G', 'B']
    per_channel = {}
    for i, name in enumerate(channel_names):
        channel_pert = abs_perturbation[:, :, i]
        per_channel[name] = {
            'max': float(np.max(channel_pert)),
            'mean': float(np.mean(channel_pert)),
            'std': float(np.std(channel_pert))
        }
    
    # SSIM
    ssim = compute_ssim(clean_img, attacked_img)
    
    # Perturbation distribution statistics
    pert_std = float(np.std(abs_perturbation))
    pert_median = float(np.median(abs_perturbation))
    
    # Percentage of perturbation in different magnitude ranges
    pert_ranges = {
        'tiny (0-0.01)': float(np.mean((abs_perturbation > 0) & (abs_perturbation <= 0.01)) * 100),
        'small (0.01-0.1)': float(np.mean((abs_perturbation > 0.01) & (abs_perturbation <= 0.1)) * 100),
        'medium (0.1-0.3)': float(np.mean((abs_perturbation > 0.1) & (abs_perturbation <= 0.3)) * 100),
        'large (>0.3)': float(np.mean(abs_perturbation > 0.3) * 100)
    }
    
    return {
        'l_inf': l_inf,
        'l_inf_255': l_inf * 255,  # In 0-255 scale for interpretability
        'l2': l2,
        'l2_normalized': l2_normalized,
        'pixels_changed': pixels_changed,
        'pixels_changed_pct': pixels_changed_pct,
        'avg_perturbation': avg_perturbation,
        'avg_perturbation_255': avg_perturbation * 255,
        'perturbation_std': pert_std,
        'perturbation_median': pert_median,
        'ssim': ssim,
        'per_channel': per_channel,
        'perturbation_ranges': pert_ranges,
        'total_pixels': total_pixels
    }


# ============================================================
# Image Loading and Parsing
# ============================================================

def load_image(path: str) -> np.ndarray:
    """Load image and convert to numpy array with values in [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img) / 255.0


def parse_attack_results(results_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Parse attack_results directory and group files by base image name.
    
    Returns:
        Dictionary mapping base_name -> {attack_type: file_path}
    """
    images = defaultdict(dict)
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.png'):
            continue
            
        # Parse filename: {base_name}_{attack_type}.png
        # e.g., "bubble.jpeg_clean.png" -> base="bubble.jpeg", attack="clean"
        parts = filename.rsplit('_', 1)
        if len(parts) != 2:
            continue
            
        base_name = parts[0]
        attack_type = parts[1].replace('.png', '')
        
        images[base_name][attack_type] = os.path.join(results_dir, filename)
    
    return dict(images)


# ============================================================
# Output Formatting
# ============================================================

def print_separator(char: str = '=', length: int = 100):
    print(char * length)


def print_image_results(base_name: str, results: Dict[str, Dict]):
    """Print formatted results for a single image."""
    print_separator()
    print(f"IMAGE: {base_name}")
    print_separator('-')
    
    # Header
    print(f"{'Attack':<8} | {'L-inf':<12} | {'L-inf(255)':<10} | {'L2 norm':<12} | "
          f"{'Pixels Changed':<18} | {'Avg Pert(255)':<12} | {'SSIM':<8}")
    print_separator('-')
    
    for attack, metrics in results.items():
        print(f"{attack:<8} | {metrics['l_inf']:<12.6f} | {metrics['l_inf_255']:<10.2f} | "
              f"{metrics['l2_normalized']:<12.6f} | "
              f"{metrics['pixels_changed']:>6} ({metrics['pixels_changed_pct']:>5.1f}%) | "
              f"{metrics['avg_perturbation_255']:<12.4f} | {metrics['ssim']:<8.4f}")
    
    print()
    
    # Per-channel breakdown for each attack
    print("Per-Channel Max Perturbation (0-1 scale):")
    print(f"{'Attack':<8} | {'R max':<10} | {'G max':<10} | {'B max':<10}")
    print_separator('-', 50)
    for attack, metrics in results.items():
        pc = metrics['per_channel']
        print(f"{attack:<8} | {pc['R']['max']:<10.6f} | {pc['G']['max']:<10.6f} | {pc['B']['max']:<10.6f}")
    print()


def print_summary(all_results: Dict[str, Dict[str, Dict]]):
    """Print summary statistics across all images for each attack type."""
    print_separator('=')
    print("SUMMARY STATISTICS BY ATTACK TYPE")
    print_separator('=')
    
    attack_types = ['fgm', 'pgd', 'hc']
    
    # Collect metrics for each attack type
    attack_metrics = {attack: [] for attack in attack_types}
    
    for base_name, results in all_results.items():
        for attack in attack_types:
            if attack in results:
                attack_metrics[attack].append(results[attack])
    
    # Print summary table
    print(f"\n{'Metric':<25} | {'FGM':<18} | {'PGD':<18} | {'HC':<18}")
    print_separator('-', 80)
    
    metric_names = [
        ('l_inf', 'L-infinity (0-1)', '{:.6f}'),
        ('l_inf_255', 'L-infinity (0-255)', '{:.2f}'),
        ('l2_normalized', 'L2 Normalized', '{:.6f}'),
        ('pixels_changed_pct', 'Pixels Changed (%)', '{:.2f}'),
        ('avg_perturbation_255', 'Avg Perturbation (255)', '{:.4f}'),
        ('ssim', 'SSIM', '{:.4f}'),
    ]
    
    for key, name, fmt in metric_names:
        values = []
        for attack in attack_types:
            if attack_metrics[attack]:
                vals = [m[key] for m in attack_metrics[attack]]
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                values.append(f"{fmt.format(mean_val)} +/- {fmt.format(std_val)}")
            else:
                values.append("N/A")
        
        print(f"{name:<25} | {values[0]:<18} | {values[1]:<18} | {values[2]:<18}")
    
    print()
    
    # Additional insights
    print_separator('-', 80)
    print("KEY OBSERVATIONS:")
    print_separator('-', 80)
    
    for attack in attack_types:
        if attack_metrics[attack]:
            avg_linf = np.mean([m['l_inf_255'] for m in attack_metrics[attack]])
            avg_pixels = np.mean([m['pixels_changed_pct'] for m in attack_metrics[attack]])
            avg_ssim = np.mean([m['ssim'] for m in attack_metrics[attack]])
            
            print(f"\n{attack.upper()}:")
            print(f"  - Average max perturbation: {avg_linf:.2f}/255 ({avg_linf/255*100:.1f}% of max range)")
            print(f"  - Average pixels affected: {avg_pixels:.1f}%")
            print(f"  - Average structural similarity: {avg_ssim:.4f}")
            
            # Characterize the attack
            if avg_pixels > 90:
                print(f"  - Characteristics: Dense perturbation affecting most pixels")
            elif avg_pixels < 20:
                print(f"  - Characteristics: Sparse perturbation affecting few pixels")
            else:
                print(f"  - Characteristics: Moderate coverage perturbation")


def save_csv(all_results: Dict[str, Dict[str, Dict]], output_path: str):
    """Save detailed results to CSV file."""
    rows = []
    
    for base_name, results in all_results.items():
        for attack, metrics in results.items():
            row = {
                'image': base_name,
                'attack': attack,
                'l_inf': metrics['l_inf'],
                'l_inf_255': metrics['l_inf_255'],
                'l2': metrics['l2'],
                'l2_normalized': metrics['l2_normalized'],
                'pixels_changed': metrics['pixels_changed'],
                'pixels_changed_pct': metrics['pixels_changed_pct'],
                'total_pixels': metrics['total_pixels'],
                'avg_perturbation': metrics['avg_perturbation'],
                'avg_perturbation_255': metrics['avg_perturbation_255'],
                'perturbation_std': metrics['perturbation_std'],
                'perturbation_median': metrics['perturbation_median'],
                'ssim': metrics['ssim'],
                'R_max': metrics['per_channel']['R']['max'],
                'G_max': metrics['per_channel']['G']['max'],
                'B_max': metrics['per_channel']['B']['max'],
                'R_mean': metrics['per_channel']['R']['mean'],
                'G_mean': metrics['per_channel']['G']['mean'],
                'B_mean': metrics['per_channel']['B']['mean'],
            }
            rows.append(row)
    
    if rows:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDetailed results saved to: {output_path}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    # Configuration
    RESULTS_DIR = "attack_results"
    OUTPUT_CSV = "perturbation_analysis.csv"
    ATTACK_TYPES = ['fgm', 'pgd', 'hc']
    
    print("=" * 100)
    print("ADVERSARIAL PERTURBATION ANALYSIS")
    print("Comparing FGM, PGD, and Hill-Climbing attacks on VGG-16")
    print("=" * 100)
    print()
    
    # Parse directory structure
    image_groups = parse_attack_results(RESULTS_DIR)
    print(f"Found {len(image_groups)} base images in {RESULTS_DIR}/")
    print(f"Images: {', '.join(sorted(image_groups.keys()))}")
    print()
    
    # Store all results
    all_results = {}
    
    # Process each image
    for base_name in sorted(image_groups.keys()):
        files = image_groups[base_name]
        
        # Check if clean version exists
        if 'clean' not in files:
            print(f"Warning: No clean version found for {base_name}, skipping...")
            continue
        
        # Load clean image
        clean_img = load_image(files['clean'])
        
        # Compute metrics for each attack type
        image_results = {}
        for attack in ATTACK_TYPES:
            if attack not in files:
                print(f"Warning: No {attack} version found for {base_name}")
                continue
            
            attacked_img = load_image(files[attack])
            metrics = compute_metrics(clean_img, attacked_img)
            image_results[attack] = metrics
        
        all_results[base_name] = image_results
        
        # Print results for this image
        print_image_results(base_name, image_results)
    
    # Print summary statistics
    print_summary(all_results)
    
    # Save to CSV
    save_csv(all_results, OUTPUT_CSV)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
