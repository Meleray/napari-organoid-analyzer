import numpy as np
import napari

# Create a viewer
viewer = napari.Viewer()

# Example image data with multiple frames (e.g., a video with 3 frames)
image_data = np.random.random((3, 512, 512))  # 3 frames of 512x512 images
viewer.add_image(image_data, name="Video")

# Example bounding boxes for each frame
# Each frame has a list of bounding boxes, where each box is defined by its corners
shapes_data = [
    np.array([[100, 100], [150, 100], [150, 150], [100, 150]]),  # Frame 0
    np.array([[200, 200], [250, 200], [250, 250], [200, 250]]),  # Frame 1
    np.array([])   # Frame 2
]

# Add the shapes layer with multiple frames
viewer.add_shapes(
    shapes_data,
    shape_type='rectangle',
    edge_color='red',
    face_color='transparent',
    name="Detections"
)

napari.run()