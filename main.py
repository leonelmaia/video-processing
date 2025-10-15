import os
import glob
import clip
import shutil
import argparse
import tempfile
import numpy as np
import gradio as gr
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from utils import (
    extract_frames,
    clean_frames,
    convert_to_browser_mp4,
    get_clip_features,
    get_features,
    reconstruct_order,
    make_video,
)
from tensorflow.keras.applications.resnet50 import ResNet50

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Reorder the mixed frames of a video given, and remove noisy frames"
    )

    parser.add_argument("--video", type=Path, help="path to video", required=True)
    return parser


def main(args):
    CLEAN_DIR = "clean_frames"
    REMOVED_DIR = "removed_frames"
    ORDER_DIR = "ordered_frames"
    EPS = 18
    MIN_SAMPLES = 5
    PCA_COMPONENTS = 0.95
    input_video = args.video
    extract_frames(input_video, "frames")
    os.makedirs("input", exist_ok=True)
    conversed_path = None
    filename = os.path.basename(input_video)

    if "conversed" not in filename:
        conversed_path = os.path.abspath(
            "/home/leonel/INA/input/" + filename.replace(".mp4", "_conversed.mp4")
        )
        convert_to_browser_mp4(input_video, output_path=conversed_path)

    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    paths = sorted(glob.glob("frames/*.jpg"))
    # device = "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)

    for folder in [CLEAN_DIR, REMOVED_DIR, ORDER_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    clean_paths, clean_features, labels = clean_frames(
        paths,
        model,
        n_clusters=2,
        pca_components=PCA_COMPONENTS,
        eps=EPS,
        min_samples=MIN_SAMPLES,
    )
    clean_features = np.array([get_features(p, model) for p in clean_paths])

    pca = PCA(n_components=PCA_COMPONENTS)
    clean_features = pca.fit_transform(clean_features)

    ordered_paths = reconstruct_order(
        clean_paths, clean_features, w_feat=1, w_motion=0, smoothing=True
    )

    for idx, p in enumerate(ordered_paths, start=1):
        filename = f"frame_{idx:04d}.jpg"
        dst_path = os.path.join(ORDER_DIR, filename)
        shutil.copy(p, dst_path)
        print(f"Copied {p} → {dst_path}")
    print(idx)

    os.makedirs("output", exist_ok=True)
    output_path = os.path.abspath("/home/leonel/INA/output/reconstructed_video.mp4")

    path, path_reversed_video = make_video(
        ordered_paths, output_path=output_path, fps=25
    )


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)
