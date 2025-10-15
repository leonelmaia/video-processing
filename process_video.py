import os
import glob
import clip
import shutil
import numpy as np
import gradio as gr
from sklearn.decomposition import PCA
from utils import (
    convert_to_browser_mp4,
    extract_frames,
    clean_frames,
    get_clip_features,
    get_features,
    reconstruct_order,
    make_video,
)
from tensorflow.keras.applications.resnet50 import ResNet50

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def process_video(input_video, w_feat=0.7, w_motion=0.2, progress=gr.Progress()):
    """
    Fully process a video to clean frames, reorder them based on features and motion,
    and generate reconstructed and reversed MP4 outputs.

    The function performs the following steps:
    1. Extracts all frames from the input video.
    2. Converts the video to a browser-compatible MP4 if not already done.
    3. Uses a pretrained CNN (ResNet50) to extract feature embeddings for each frame.
    4. Cleans the frames by removing outliers using PCA + DBSCAN clustering.
    5. Reconstructs an optimized frame order based on feature similarity and motion.
    6. Copies the reordered frames into an output folder.
    7. Generates a reconstructed MP4 video and a reversed version for playback.

    Args:
        input_video (str):
            Path to the input video file to process.
        w_feat (float, optional):
            Weight for feature similarity when reconstructing frame order. Defaults to 0.7.
        w_motion (float, optional):
            Weight for motion consistency when reconstructing frame order. Defaults to 0.2.
        progress (gr.Progress, optional):
            Optional progress indicator object (from Gradio) to report processing progress.

    Returns:
        tuple[str, str]:
            - Path to the reconstructed video (browser-compatible MP4).
            - Path to the reversed reconstructed video.

    """

    CLEAN_DIR = "clean_frames"
    REMOVED_DIR = "removed_frames"
    ORDER_DIR = "ordered_frames"
    EPS = 18
    MIN_SAMPLES = 5
    PCA_COMPONENTS = 0.95

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
        clean_paths, clean_features, w_feat=w_feat, w_motion=w_motion, smoothing=True
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

    if not os.path.exists(path):
        raise FileNotFoundError(f"Output video not found: {output_path}")
    print(f"Reconstructed video saved to {output_path}")
    return path, path_reversed_video


with gr.Blocks() as demo:
    gr.Markdown("### Video Reordering Demo")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                w_feat_input = gr.Slider(
                    0, 1, value=1, step=0.05, label="Feature Weight"
                )
                w_motion_input = gr.Slider(
                    0, 1, value=0, step=0.05, label="Motion Weight"
                )
            input_vid = gr.Video(label="Upload Video")
            submit_btn = gr.Button("Process Video")
        with gr.Column():
            output_vid1 = gr.Video(label="Processed Video")
            output_vid2 = gr.Video(label="Reversed Video")

    def update_motion(w_feat):
        w_feat = min(max(w_feat, 0), 1)
        return round(1 - w_feat, 2)

    def update_feat(w_motion):
        w_motion = min(max(w_motion, 0), 1)
        return round(1 - w_motion, 2)

    w_feat_input.release(update_motion, inputs=w_feat_input, outputs=w_motion_input)
    w_motion_input.release(update_feat, inputs=w_motion_input, outputs=w_feat_input)

    submit_btn.click(
        process_video,
        inputs=[
            input_vid,
            w_feat_input,
            w_motion_input,
        ],
        outputs=[output_vid1, output_vid2],
    )

    gr.Examples(
        examples=[
            ["input/corrupted_video_conversed.mp4", 1, 0],
            ["input/corrupted_video2_conversed.mp4", 1, 0],
            ["input/corrupted_video3_conversed.mp4", 1, 0],
        ],
        inputs=[input_vid, w_feat_input, w_motion_input],
        label="Example Videos",
    )


if __name__ == "__main__":
    demo.launch()
