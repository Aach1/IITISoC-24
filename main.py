import cv2
import numpy as np
import torch
from torchvision import models, transforms
# from PIL import Image

# Load the DeepLabV3 model from torchvision
replace_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Define the transform to preprocess the frames
preprocess = transforms.Compose([
    transforms.ToPILImage(),  # converts a NumPy array or a tensor into a PIL image
    transforms.Resize((513, 513)),  # resizes input image to desired size
    transforms.ToTensor(),  # converts a PIL Image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizes the tensor with mean and sd values for each channel
])


# Helper function to process frames of video to replace bg
def process_frame(frame, background_image):
    # Convert frame to tensor and preprocess
    input_tensor = preprocess(frame)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Run the model
    with torch.no_grad():
        output = replace_model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)

    # Resize mask to match the original frame size
    mask = output_predictions.byte().cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Create a binary mask for the background and foreground
    mask = mask == 15  # Assuming class 15 is 'person'
    mask = mask.astype(np.uint8)

    # Replace the background
    foreground = frame * mask[:, :, np.newaxis]
    background = background_image * (1 - mask[:, :, np.newaxis])
    output_frame = foreground + background

    return output_frame


# Load the input video
# input_video_path = input("Input video path: ")
input_video_path = ""  # "C:/Users/Aachi/Downloads/gymnast2.mp4"
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Load the background image and resize it to match video frames
# background_image_path = input("Background Image path: ")
background_image_path = ""  # "C:/Users/Aachi/Downloads/white.jpg"
background_image = cv2.imread(background_image_path)
background_image = cv2.resize(background_image, (width, height))

# Define the codec and create VideoWriter object
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = process_frame(frame, background_image)
    out.write(np.uint8(output_frame))

    # Display the resulting frame
    cv2.imshow('Output Video', np.uint8(output_frame))

    # Press 'q' to exit the video display window early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
