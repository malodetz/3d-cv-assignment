import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from typing import Union
from pathlib import Path


def read_image(file_path: Union[str, Path]) -> np.ndarray:
    """
    Reads an image from the specified file path.

    Parameters:
        file_path (Union[str, Path]): Path to the image file.

    Returns:
        numpy.ndarray: The image in BGR format.
    """
    return cv2.imread(str(file_path), cv2.IMREAD_COLOR)


def show_image(image: np.ndarray) -> None:
    """
    Displays the image.

    Parameters:
        image (numpy.ndarray): The image array in BGR format.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def read_lidar(file_path: Union[str, Path]) -> np.ndarray:
    """
    Reads a LiDAR point cloud from the specified file path.

    Parameters:
        file_path (Union[str, Path]): Path to the LiDAR file.

    Returns:
        numpy.ndarray: Nx4 array of LiDAR points (X, Y, Z, Intensity).
    """
    scan = np.fromfile(str(file_path), dtype=np.float32)
    return scan.reshape((-1, 5))[:, :4]


def show_lidar_pointcloud(points: np.ndarray, title: str = "LiDAR Point Cloud") -> None:
    """
    Displays the LiDAR point cloud interactively in 3D.

    Parameters:
        points (numpy.ndarray): Nx3 or Nx4 array of LiDAR points (X, Y, Z, [Intensity]).
        title (str, optional): Title of the plot.
    """
    color = points[:, 3] if points.shape[1] == 4 else points[:, 2]  # Use intensity or Z for color

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],  # X-coordinates
                y=points[:, 1],  # Y-coordinates
                z=points[:, 2],  # Z-coordinates
                mode='markers',
                marker=dict(
                    size=2,                 # Size of points
                    color=color,            # Color based on Z or intensity
                    colorscale='Viridis',   # Colormap
                    opacity=0.8             # Transparency
                )
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)",
            aspectmode="auto"
        )
    )

    fig.show()