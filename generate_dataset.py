import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# Import your custom background generator
# (Ensure your previous script is saved as mvn_lumpy.py in the same directory)
from mvn_lumpy import mvn_type_2_lumpy 

# Assuming you still want to use the same signal generator from your reference
from signal_present import disk_signal 

np.random.seed(0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simulated medical image datasets using MVN Lumpy backgrounds."
    )

    # Required positional arguments
    parser.add_argument(
        "outdir",
        default="./data/",
        help="Base output folder where datasets will be saved. Default: current directory",
    )
    parser.add_argument(
        "num_datasets",
        type=int,
        help="Number of datasets to be generated (required)",
    )
    parser.add_argument(
        "image_dim",
        type=int,
        help="Image dimension (e.g., 64 for 64x64) (required)",
    )
    parser.add_argument(
        "lump_width",
        type=float,
        help="Lump width for the MVN Lumpy background (e.g., 5.0) (required)",
    )

    # Dataset generation flags
    parser.add_argument(
        "--absent",
        action="store_true",
        help="Generate signal absent images",
    )
    parser.add_argument(
        "--present",
        action="store_true",
        help="Generate signal present images",
    )

    # Optional parameters for background and signal
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.07,
        help="Alpha value for the disk signal (float, default: 0.07)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Radius value for the disk signal (float, default: 2.0)",
    )

    return parser.parse_args()

def save_config(args):
    """
    Save the command-line arguments to <outdir>/config.txt.
    """
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    config_path = os.path.join(outdir, "config.txt")
    with open(config_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key} = {value}\n")

def to_uint8_image(img_array):
    """
    Safely rescales a float array to 0-255 and converts it to uint8.
    This prevents underflow/overflow artifacts when saving normal distributions.
    """

    M = 0.0 # depends on muimg
    S = 1.0 # depends on sigimg
    std_dev_range = 4.0 # how many standard deviations to include

    # Define fixed bounds
    fixed_min = M - (std_dev_range * S)
    fixed_max = M + (std_dev_range * S)

    # Perform fixed normalization
    # Formula: (val - min) / (max - min) * 255
    img_normalized = (img_array - fixed_min) / (fixed_max - fixed_min) * 255

    # Clip values to ensure they stay in [0, 255] before converting to uint8
    img_uint8 = np.clip(img_normalized, 0, 255).astype(np.uint8)

    return img_uint8

def run(data_path: str, num_datasets: int, image_dim: int, lump_width: float, alpha: float, radius: float, generate_absent: bool, generate_present: bool) -> None:
    print("========================================")
    print(f"Generating {num_datasets} datasets in {data_path}")
    print(f"Image dimension: {image_dim}x{image_dim}")
    print(f"Lump Width (Background): {lump_width}")
    print(f"Signal Alpha: {alpha} | Signal Radius: {radius}")
    print(f"Generate signal absent images: {generate_absent}")
    print(f"Generate signal present images: {generate_present}")
    print("========================================")

    # Create subfolders
    os.makedirs(data_path, exist_ok=True)
    if generate_absent:
        absent_clean_png_dir = os.path.join(data_path, "signal_absent_clean_png")
        os.makedirs(absent_clean_png_dir, exist_ok=True)
        absent_clean_npy_dir = os.path.join(data_path, "signal_absent_clean_npy")
        os.makedirs(absent_clean_npy_dir, exist_ok=True)
        absent_noise_png_dir = os.path.join(data_path, "signal_absent_noise_png")
        os.makedirs(absent_noise_png_dir, exist_ok=True)
        absent_noise_npy_dir = os.path.join(data_path, "signal_absent_noise_npy")
        os.makedirs(absent_noise_npy_dir, exist_ok=True)
    if generate_present:
        present_clean_png_dir = os.path.join(data_path, "signal_present_clean_png")
        os.makedirs(present_clean_png_dir, exist_ok=True)
        present_clean_npy_dir = os.path.join(data_path, "signal_present_clean_npy")
        os.makedirs(present_clean_npy_dir, exist_ok=True)
        present_noise_png_dir = os.path.join(data_path, "signal_present_noise_png")
        os.makedirs(present_noise_png_dir, exist_ok=True)
        present_noise_npy_dir = os.path.join(data_path, "signal_present_noise_npy")
        os.makedirs(present_noise_npy_dir, exist_ok=True)

    # Base inputs for the MVN Lumpy function
    muimg = np.zeros((image_dim, image_dim), dtype=np.float64)
    sigimg = np.ones((image_dim, image_dim), dtype=np.float64)

    # Pre-generate the signal (assuming the signal is identical across the dataset)
    if generate_present:
        signal = disk_signal(image_dim, alpha, radius)

    ''' Generate datasets '''
    for i in tqdm(range(num_datasets), total=num_datasets, desc="Generating datasets"):
        
        # 1. Generate the base lumpy background
        # (This acts as our 'absent' base)
        img_bg = mvn_type_2_lumpy(lump_width, muimg, sigimg, numimgs=1)
        
        ''' Signal Absent Image Generation '''
        if generate_absent:
            img_absent_noise = img_bg + 0.1 * np.random.randn(*img_bg.shape)

            # Save arrays
            np.save(os.path.join(absent_clean_npy_dir, f"lumpy_{i}.npy"), img_bg)
            np.save(os.path.join(absent_noise_npy_dir, f"lumpy_noise_{i}.npy"), img_absent_noise)

            # Save standard images safely
            cv2.imwrite(os.path.join(absent_clean_png_dir, f"lumpy_{i}.png"), to_uint8_image(img_bg))
            cv2.imwrite(os.path.join(absent_noise_png_dir, f"lumpy_noise_{i}.png"), to_uint8_image(img_absent_noise))
        
        ''' Signal Present Image Generation '''
        if generate_present:
            img_present = img_bg + signal
            img_present_noise = img_present + 0.1 * np.random.randn(*img_present.shape)

            # Save arrays
            np.save(os.path.join(present_clean_npy_dir, f"lumpy_{i}.npy"), img_present)
            np.save(os.path.join(present_noise_npy_dir, f"lumpy_noise_{i}.npy"), img_present_noise)

            # Save standard images safely
            cv2.imwrite(os.path.join(present_clean_png_dir, f"lumpy_{i}.png"), to_uint8_image(img_present))
            cv2.imwrite(os.path.join(present_noise_png_dir, f"lumpy_noise_{i}.png"), to_uint8_image(img_present_noise))


if __name__ == "__main__":
    """
    Usage:
        python generate_dataset.py <outdir> <num_datasets> <image_dim> <lump_width> [--alpha ALPHA] [--radius RADIUS] [--absent] [--present]
    
    Example:
        python generate_dataset.py ./data/ 20000 64 5.0 --alpha 0.3 --radius 2.0 --absent --present
    """

    args = parse_args()
    save_config(args)
    
    run(
        data_path=args.outdir, 
        num_datasets=args.num_datasets, 
        image_dim=args.image_dim, 
        lump_width=args.lump_width,
        alpha=args.alpha, 
        radius=args.radius, 
        generate_absent=args.absent, 
        generate_present=args.present
    )
