import os
import cv2
import numpy as np
from tqdm import tqdm
from nuscenes import NuScenes
from pyquaternion import Quaternion
from ultralytics import YOLO

class LidarProjector:
    def __init__(self, nusc, sample_data):
        self.calibrated_sensor = nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        self.sensor_rotation = Quaternion(self.calibrated_sensor["rotation"])
        self.sensor_translation = np.array(self.calibrated_sensor["translation"])

        self.intrinsic = np.array(self.calibrated_sensor["camera_intrinsic"])
        self.ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        self.ego_rotation = Quaternion(self.ego_pose["rotation"])
        self.ego_translation = np.array(self.ego_pose["translation"])

    def project(self, points):
        # Transform from global to ego vehicle coordinates
        points = points - self.ego_translation
        rotated_points_ego = np.array(
            [self.ego_rotation.inverse.rotate(point) for point in points]
        )

        # Transform from ego vehicle to camera coordinates
        points_cam = rotated_points_ego - self.sensor_translation
        # Use correct rotation direction (not inverse)
        rotated_points_sensor = np.array(
            [self.sensor_rotation.rotate(point) for point in points_cam]
        )

        # Filter points in front of the camera (positive z-axis in camera coordinates)
        valid_points = rotated_points_sensor[rotated_points_sensor[:, 2] > 0]
        if valid_points.shape[0] == 0:  # Check if any points are left after filtering
            return np.empty((0, 2), dtype=int)  # Return empty array if no points

        # Project 3D points to 2D image plane
        projected = self.intrinsic @ valid_points.T
        projected = projected / projected[2, :]
        return projected[:2].T.astype(int)


class NuScenesProcessor:
    def __init__(self, dataroot="./v1.0-mini"):
        self.nusc = NuScenes(version="v1.0-mini", dataroot=dataroot)
        self.model = YOLO("yolo11n.pt")

    def process_scene(self, scene_index=0):
        scene = self.nusc.scene[scene_index]
        samples = [
            sample["token"]
            for sample in self.nusc.sample
            if sample["scene_token"] == scene["token"]
        ]

        # Initialize video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output.mp4", fourcc, 2, (1600, 900))

        for sample_token in tqdm(samples):
            # Get camera and lidar data
            sample_info = self.nusc.get("sample", sample_token)
            cam_data = self.nusc.get("sample_data", sample_info["data"]["CAM_FRONT"])
            lidar_data = self.nusc.get("sample_data", sample_info["data"]["LIDAR_TOP"])

            # Process image
            img = self.load_image(cam_data)
            detections = self.detect_objects(img)

            # Load LiDAR points
            lidar_points = self.load_lidar(lidar_data)

            # Get LiDAR calibration and pose data
            lidar_calibrated = self.nusc.get(
                "calibrated_sensor", lidar_data["calibrated_sensor_token"]
            )
            lidar_ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

            # Transform LiDAR points to ego vehicle coordinates
            lidar_rotation = Quaternion(lidar_calibrated["rotation"])
            lidar_translation = np.array(lidar_calibrated["translation"])
            points_ego = np.array(
                [
                    lidar_rotation.rotate(point[:3]) + lidar_translation
                    for point in lidar_points
                ]
            )

            # Transform from ego to global coordinates
            ego_rotation = Quaternion(lidar_ego_pose["rotation"])
            ego_translation = np.array(lidar_ego_pose["translation"])
            points_global = np.array(
                [ego_rotation.rotate(point) + ego_translation for point in points_ego]
            )

            # Project LiDAR points to camera image
            projector = LidarProjector(self.nusc, cam_data)
            projected = projector.project(points_global)

            # Filter points inside the image
            valid_points = self.filter_points(img.shape, projected)

            # Draw results
            img = self.draw_results(
                img, detections, valid_points, projected, points_global
            )
            out.write(img)

        out.release()

    def load_image(self, sample_data):
        path = os.path.join(self.nusc.dataroot, sample_data["filename"])
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_lidar(self, sample_data):
        path = os.path.join(self.nusc.dataroot, sample_data["filename"])
        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]

    def detect_objects(self, img):
        results = self.model(img)[0]
        return [
            {
                "box": box.xyxy[0].cpu().numpy(),
                "cls": self.model.names[int(box.cls)],
                "conf": float(box.conf),
            }
            for box in results.boxes
        ]

    def filter_points(self, img_shape, points):
        mask = (
            (points[:, 0] >= 0)
            & (points[:, 0] < img_shape[1])
            & (points[:, 1] >= 0)
            & (points[:, 1] < img_shape[0])
        )
        return points[mask]

    def draw_results(self, img, detections, points, projected, original_3d_points):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Create detection boxes structure
        det_boxes = [(det["box"].astype(int), det["cls"]) for det in detections]

        # Create point mapping and filter points in boxes
        points_in_any_box = set()
        point_box_mapping = {}

        # First pass: Find all points in any detection box
        for p_idx, p in enumerate(points):
            for box_idx, (box, _) in enumerate(det_boxes):
                x1, y1, x2, y2 = box
                if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                    points_in_any_box.add(p_idx)
                    point_box_mapping.setdefault(p_idx, []).append(box_idx)
                    break  # No need to check other boxes once matched

        # Draw LiDAR points (only those in boxes)
        for p_idx in points_in_any_box:
            cv2.circle(img, tuple(points[p_idx]), 2, (0, 255, 0), -1)

        # Draw detections and calculate distances
        for det_idx, (box, cls) in enumerate(det_boxes):
            x1, y1, x2, y2 = box
            associated_points = []

            # Find points specifically in this box
            for p_idx in points_in_any_box:
                if det_idx in point_box_mapping.get(p_idx, []):
                    associated_points.append(original_3d_points[p_idx])

            # Draw detection info
            if associated_points:
                distances = [np.linalg.norm(p) for p in associated_points]
                median_dist = np.median(distances)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"{cls}: {median_dist/100:.2f}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        return img


if __name__ == "__main__":
    processor = NuScenesProcessor()
    processor.process_scene()
