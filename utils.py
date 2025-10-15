import os
import cv2
import torch
import shutil
import subprocess
import numpy as np
import imageio_ffmpeg
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import preprocess_input

# Graph imports
import networkx as nx
from networkx.algorithms import approximation as approx


def extract_frames(video_path, output_folder):
    """
    Extract all frames from a video and save them as individual image files.

    This function reads a video from the specified path, removes any existing
    output folder to avoid conflicts, creates a new folder, and writes each
    frame as a separate image (JPEG) in the output directory. The frame count
    and video FPS are printed for reference.

    Args:
        video_path (str):
            Path to the input video file.
        output_folder (str):
            Path to the folder where extracted frames will be saved. If the folder
            already exists, it will be deleted and recreated.

    Returns:
        None
    """

    if os.path.exists(output_folder):
        print(f"Removing existing folder: {output_folder}")
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/{count:04d}.jpg", frame)
        count += 1
    cap.release()
    print(f"{count} frames extraites.")


def get_features(img_path, model):
    """
    Extract and return a flattened feature embedding from an image using a pretrained CNN.

    This function loads an image from the specified path, resizes it to 224x224 pixels,
    converts it to an array, preprocesses it according to the model's requirements,
    and then passes it through the model to obtain a feature embedding. The embedding
    is returned as a flattened 1D NumPy array.

    Args:
        img_path (str):
            Path to the input image file.
        model (keras.Model):
            Pretrained CNN model (e.g., ResNet50 with `include_top=False` and pooling='avg')
            used to extract feature embeddings.

    Returns:
        np.ndarray:
            Flattened feature vector representing the input image.

    Notes:
        - The preprocessing step uses `preprocess_input` (from Keras applications)
          to normalize image pixels according to the model’s expected input.
        - The resulting feature vector can be used for clustering, similarity
          search, or other downstream computer vision tasks.

    Example:
        >>> feat = get_features("frame_001.jpg", model)
        >>> print(feat.shape)
        (2048,)  # Example dimension depending on the CNN
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()


def get_clip_features(img_path, model, preprocess, device="cpu"):
    """
    Extract normalized feature embeddings from an image using a CLIP model.

    This function loads an image from a given file path, applies the provided
    preprocessing pipeline, and computes its feature embedding using a CLIP model.
    The embedding is L2-normalized and returned as a flattened NumPy array.

    Args:
        img_path (str):
            Path to the input image file.
        model (torch.nn.Module):
            Pretrained CLIP model with an `encode_image` method.
        preprocess (callable):
            Preprocessing function or pipeline compatible with the CLIP model
            (e.g., resizing, normalization, tensor conversion).
        device (str, optional):
            Device on which to perform computation, e.g., `"cpu"` or `"cuda"`.
            Defaults to `"cpu"`.

    Returns:
        np.ndarray:
            L2-normalized feature embedding of the input image, flattened to a 1D array.

    Notes:
        - The function uses `torch.no_grad()` to avoid computing gradients for efficiency.
        - L2 normalization ensures embeddings are comparable using cosine similarity.
        - Flattening produces a 1D vector suitable for clustering, similarity search, or indexing.
    """
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    return emb.cpu().numpy().flatten()


def clean_frames(
    paths, model, n_clusters=2, pca_components=0.95, eps=0.5, min_samples=5
):
    """
    Detect and remove visually inconsistent (outlier) frames using deep feature clustering.

    The function extracts high-level visual embeddings for each frame using a pretrained
    CNN model, reduces dimensionality with PCA, and applies DBSCAN clustering to identify
    outlier frames. Frames labeled as noise (`-1` by DBSCAN) are considered inconsistent
    and are copied to a `removed_frames` directory for inspection.

    Args:
        paths (list[str]):
            List of file paths to frame images.
        model (keras.Model or torch.nn.Module):
            Pretrained feature extractor model (e.g., ResNet50 with `pooling='avg'`).
        n_clusters (int, optional):
            Unused parameter reserved for compatibility with previous KMeans-based
            implementations. Defaults to 2.
        pca_components (float or int, optional):
            Number of PCA components to retain. If float (0 < value ≤ 1),
            it represents the fraction of variance to preserve. Defaults to 0.95.
        eps (float, optional):
            Maximum distance between two samples for them to be considered neighbors
            in DBSCAN. Controls cluster tightness. Defaults to 0.5.
        min_samples (int, optional):
            Minimum number of samples required in a neighborhood for a point to be
            considered a core point in DBSCAN. Defaults to 5.

    Returns:
        tuple[list[str], np.ndarray, np.ndarray]:
            - **clean_paths**: List of file paths for frames retained after cleaning.
            - **clean_embeddings**: Feature embeddings corresponding to retained frames.
            - **clean_labels**: Cluster labels assigned to retained frames.

    Notes:
        - DBSCAN is robust to noise and can discover clusters of arbitrary shape.
        - Increasing `eps` will merge nearby clusters; decreasing it makes clustering
        stricter and may flag more frames as outliers.
        - `pca_components` helps denoise feature space before clustering.
    """
    features = np.array([get_features(p, model) for p in paths])
    pca = PCA(n_components=pca_components)
    features_reduced = pca.fit_transform(features)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(features_reduced)

    clean_paths = [p for p, l in zip(paths, labels) if l != -1]
    removed_paths = [p for p, l in zip(paths, labels) if l == -1]

    for p in removed_paths:
        shutil.copy(p, os.path.join("removed_frames", os.path.basename(p)))
    pca = PCA(n_components=pca_components)

    print(f"{len(clean_paths)} frames kept out of {len(paths)}")

    mask = labels != -1
    clean_embeddings = features[mask]
    clean_labels = labels[mask]

    return clean_paths, clean_embeddings, clean_labels


def find_start_frame(features, frames, alpha=0.8, beta=0.1, gamma=0.1):
    """
    Estimate the most likely starting frame in a shuffled or unordered video sequence.

    The function computes three complementary cues for each frame:
    1. **Visual distinctiveness** — measures how different a frame's feature embedding
       is from all others (using cosine similarity).
    2. **Motion difference** — measures pixel-level frame-to-frame intensity changes.
    3. **Color histogram variance** — quantifies color distribution variability.

    These cues are normalized and linearly combined with tunable weights (`alpha`,
    `beta`, `gamma`) to produce a final score. The frame with the highest score is
    selected as the most probable starting point.

    Args:
        features (np.ndarray):
            Array of shape (n, d) containing extracted feature embeddings
            (e.g., from a CNN backbone such as ResNet) for each frame.
        frames (list[np.ndarray]):
            List of `n` video frames as NumPy arrays in BGR format.
        alpha (float, optional):
            Weight for visual distinctiveness term. Defaults to 0.8.
        beta (float, optional):
            Weight for inverse motion difference term. Defaults to 0.1.
        gamma (float, optional):
            Weight for color histogram variance term. Defaults to 0.1.

    Returns:
        tuple[int, np.ndarray]:
            - **start_frame_index**: Index of the frame with the highest combined score.
            - **start_score**: Normalized score array of length `n` for all frames.

    Notes:
        - A higher `alpha` emphasizes semantic uniqueness (appearance-based).
        - A higher `beta` penalizes strong motion (favoring stable starting frames).
        - A higher `gamma` favors frames with richer or more varied color content.

    Example:
        >>> idx, scores = find_start_frame(features, frames, alpha=0.7, beta=0.2, gamma=0.1)
        >>> print(f"Predicted first frame: {idx}")
    """
    n = len(frames)
    S = cosine_similarity(features)
    visual_score = 1 - S.mean(axis=1)

    motion_score = np.zeros(n)
    for i in range(n - 1):
        diff = np.mean(
            np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32))
        )
        motion_score[i + 1] = diff
    motion_score = (motion_score - motion_score.min()) / (
        motion_score.max() - motion_score.min() + 1e-8
    )

    color_score = np.zeros(n)
    for i, f in enumerate(frames):
        hist = cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        color_score[i] = np.std(hist)
    color_score = (color_score - color_score.min()) / (
        color_score.max() - color_score.min() + 1e-8
    )

    start_score = alpha * visual_score + beta * (1 - motion_score) + gamma * color_score
    start_frame_index = np.argmax(start_score)

    print(f"Best start frame index: {start_frame_index}")
    return start_frame_index, start_score


def smooth_order(path, D, window=5):
    """
    This function refines a given frame order (e.g., from a Traveling Salesman
    Problem solution) by performing local swaps within a sliding window.
    The goal is to minimize short-term discontinuities based on a distance
    matrix, typically representing motion or feature dissimilarity between
    frames.

    Parameters:
        path (list of int): current frame order (indices)
        D (np.array): distance matrix (n x n)
        window (int): number of frames before/after to consider for local swaps
    Returns:
        smoothed path (list of int)
    """
    n = len(path)
    smoothed_path = path.copy()

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)

        segment = smoothed_path[start:end]
        improved = True
        while improved:
            improved = False
            for j in range(len(segment) - 1):
                dist_orig = D[segment[j], segment[j + 1]]

                # distance after swap
                segment[j], segment[j + 1] = segment[j + 1], segment[j]
                dist_new = D[segment[j], segment[j + 1]]

                if dist_new < dist_orig:
                    improved = True
                else:
                    segment[j], segment[j + 1] = segment[j + 1], segment[j]

        smoothed_path[start:end] = segment

    return smoothed_path


def motion_diff(f1, f2):
    """Compute mean absolute difference between two frames."""
    return np.mean(np.abs(f1.astype(np.float32) - f2.astype(np.float32)))


def motion_finetune(path, frames, window=5):
    """
    Perform motion-based local refinement of frame order.

    This function attempts to locally reorder video frames to reduce abrupt
    motion transitions between consecutive frames. Within a defined window
    around each frame, it performs local swaps that minimize sudden changes
    in motion magnitude (measured as pixel-wise frame differences).

    Args:
        path (list[int]):
            Current frame order as a list of frame indices.
        frames (list[np.ndarray]):
            Preloaded frames corresponding to the indices in `path`.
        window (int, optional):
            Number of neighboring frames (before and after) to consider for
            local swap optimization. Default is 5.

    Returns:
        list[int]:
            Refined frame order (indices) with smoother apparent motion between
            consecutive frames.
    """
    n = len(path)
    refined_path = path.copy()

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        segment = refined_path[start:end]
        improved = True

        while improved:
            improved = False
            for j in range(len(segment) - 1):
                f1a = frames[segment[j]]
                f2a = frames[segment[j + 1]]
                dist_orig = motion_diff(f1a, f2a)

                # try swapping
                segment[j], segment[j + 1] = segment[j + 1], segment[j]
                f1b = frames[segment[j]]
                f2b = frames[segment[j + 1]]
                dist_new = motion_diff(f1b, f2b)

                if dist_new < dist_orig:
                    improved = True
                else:
                    # revert if not better
                    segment[j], segment[j + 1] = segment[j + 1], segment[j]

        refined_path[start:end] = segment

    return refined_path


def reconstruct_order(
    clean_paths, clean_features, w_feat=1, w_motion=0, w_color=0, smoothing=True
):
    """
    Create a video from a sequence of images and optionally generate a reversed version.

    This function reads a list of image files, compiles them into an MP4 video
    at the specified frame rate, and saves it to the output path. For browser
    compatibility, the video is converted to H.264 codec with YUV420p pixel format.
    Optionally, a reversed video is generated by writing the frames in reverse order.

    Args:
        image_paths (list[str]):
            List of paths to input images to be compiled into a video.
        output_path (str):
            Path where the generated video will be saved.
        fps (int, optional):
            Frame rate for the output video. Defaults to 25 frames per second.
        reversed (bool, optional):
            Whether to create a reversed version of the video. Defaults to True.

    Returns:
        tuple[str, str]:
            - Path to the generated video (`output_path` after conversion).
            - Path to the reversed video (if `reversed=True`). Note that this will
              be undefined if `reversed=False`.
    """
    frames = [cv2.imread(p) for p in clean_paths]
    start_frame_index, _ = find_start_frame(
        clean_features, frames, alpha=w_feat, beta=w_motion
    )
    n = len(clean_paths)
    D_feat = 1 - cosine_similarity(clean_features)
    D_motion = np.zeros((n, n), dtype=np.float32)
    if w_motion > 0:
        frames_array = np.array([f.astype(np.float32) for f in frames])  # (n, H, W, C)
        diff = np.abs(
            frames_array[:, None] - frames_array[None, :]
        )  # shape: (n, n, H, W, C)
        D_motion = diff.mean(axis=(2, 3, 4))

    D_feat = np.nan_to_num(D_feat, nan=0.0, posinf=0.0, neginf=0.0)
    D_motion = np.nan_to_num(D_motion, nan=0.0, posinf=0.0, neginf=0.0)

    D_feat /= D_feat.max()
    max_val = D_motion.max()
    if max_val > 0:
        D_motion /= max_val
    max_val = D_feat.max()
    if max_val > 0:
        D_feat /= max_val
    D = w_feat * D_feat + w_motion * D_motion

    G = nx.complete_graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            G[i][j]["weight"] = D[i, j]

    path = approx.greedy_tsp(G, source=start_frame_index, weight="weight")
    # path = simulated_annealing_tsp(G,source=start_frame_index, init_cycle=path, weight='weight')
    # path = reorder_frames(clean_features)

    if smoothing:
        path = smooth_order(path, D, window=5)
        path = motion_finetune(path, frames, window=5)
    ordered_paths = [clean_paths[i] for i in path]
    print(f"TSP ordering done, {len(path)} frames.")
    return ordered_paths


def make_video(image_paths, output_path, fps=25, reversed=True):
    frame = cv2.imread(image_paths[0])
    h, w, _ = frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for img_path in image_paths:
        out.write(cv2.imread(img_path))
    out.release()
    path = convert_to_browser_mp4(output_path, output_path="output/converted_video.mp4")
    if reversed:
        reversed_path = output_path.replace(".mp4", "_reversed.mp4")
        out_rev = cv2.VideoWriter(
            reversed_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for img_path in image_paths[::-1]:
            frame = cv2.imread(img_path)
            out_rev.write(frame)
        out_rev.release()
        path_reversed_video = convert_to_browser_mp4(
            reversed_path, output_path="output/converted_reversed_video.mp4"
        )
    return path, path_reversed_video


def convert_to_browser_mp4(input_path, output_path="output/converted_video.mp4"):
    """
    Convert a video to a browser-compatible MP4 format using H.264 encoding.

    This function uses `ffmpeg` (via `imageio-ffmpeg`) to convert any input video
    to an MP4 file with H.264 codec and YUV420p pixel format, ensuring maximum
    compatibility with web browsers. Existing files at the output path are
    overwritten.

    Args:
        input_path (str):
            Path to the input video file to be converted.
        output_path (str, optional):
            Path where the converted MP4 video will be saved. Defaults to
            "output/converted_video.mp4".

    Returns:
        str:
            Path to the converted MP4 video file.
    """
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return output_path
