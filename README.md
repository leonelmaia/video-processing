# Video Frame Cleaning and Reordering Project

This repository contains a project that focuses on reconstructing an original video from a shuffled and corrupted frame sequence.

## Project Overview

A random segment of a source video was selected.

The frames of this segment were shuffled, and unrelated images were inserted.

A new video was created from this disordered set of images.

The goal of this project is to develop an algorithm that can reconstruct the original video order, even when additional unrelated frames have been inserted. The algorithm is designed to be generalizable to other videos that have undergone a similar treatment.

## Key Features

Frame extraction and reconstruction using deep feature embeddings (e.g., ResNet, CLIP).

Detection and removal of outlier frames.

Frame reordering based on visual similarity, motion consistency, and/or color histograms.

Optional generation of reversed video sequences.

Fully documented and modular code for easy experimentation.

## Usage

The repository includes scripts to:

Extract frames from a video.

Clean frames and remove unrelated images.

Compute features for each frame.

Reconstruct the most likely original frame order.

Generate reconstructed videos (including reversed versions).

## Tech Stack & Libraries

Python 3.12

OpenCV, NumPy, scikit-learn, TensorFlow/Keras, PyTorch ...

CLIP for deep visual embeddings

Gradio (optional) for progress monitoring and UI

FFMPEG (via imageio-ffmpeg) for video encoding

## Goal for the Technical Interview

The objective is to demonstrate your algorithm and justify your design choices. The code should be self-contained and runnable for review before the interview.


## How to use ?

1. Create and activate a virtual environment
```
# Create a venv
python -m venv venv  

# Activate it
source venv/bin/activate 
 ```

 2. Install requirements
 ```
pip install -r requirements.txt
 ```

3. run this command
```
python process_video.py
 ``` 

Then, access : http://127.0.0.1:7860/


4. OPTIONAL
```
python main.py --video corrupted_video.mp4
 ```
 
