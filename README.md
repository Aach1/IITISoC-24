# IITISoC-24

# Video Background Replacement using DeepLabV3 and Streamlit

This project is a Streamlit application that allows users to replace the background of a video with a custom background image using a pre-trained DeepLabV3 model for segmentation.

# Overview

The application uses a pre-trained DeepLabV3 model from PyTorch to segment the foreground (person) from the background in a video. The segmented foreground is then combined with a user-provided background image to generate a new video with the replaced background.

# Features

- Upload a video and a background image.
- Replace the video background with the uploaded background image.
- Download and view the processed video directly in the browser.

# Required libraries
-streamlit
-opencv (python)
-torch
-torchvision
-numpy
-pillow

