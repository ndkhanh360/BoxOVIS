
import sys
sys.path.append("..")
from models.Mask3D.mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, map_output_to_pointcloud_no_ranking, save_colorized_mesh 
import torch
import numpy as np

def project_boxes_2d_to_3d(boxes_2d, pointcloud, projection_matrix, min_points=10):
    """
    Project 2D bounding boxes to 3D point cloud space.
    
    Args:
        boxes_2d: numpy array of shape (N, 4) containing [x1, y1, x2, y2] coordinates
        pointcloud: numpy array of shape (P, 3) containing 3D points
        projection_matrix: numpy array of shape (3, 4) camera projection matrix
        min_points: minimum number of points required in a box to consider it valid
    
    Returns:
        boxes_3d: list of (8, 3) arrays containing 3D box corners
        point_masks: list of boolean masks indicating which points belong to each box
        valid_boxes: boolean array indicating which boxes are valid (have enough points)
    """
    if isinstance(boxes_2d, torch.Tensor):
        boxes_2d = boxes_2d.cpu().numpy()
    if isinstance(pointcloud, torch.Tensor):
        pointcloud = pointcloud.cpu().numpy()
    if isinstance(projection_matrix, torch.Tensor):
        projection_matrix = projection_matrix.cpu().numpy()

    # Project all 3D points to 2D
    points_homo = np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1))], axis=1)
    points_2d_homo = (projection_matrix @ points_homo.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # normalize by depth

    boxes_3d = []
    point_masks = []
    valid_boxes = []

    for box in boxes_2d:
        x1, y1, x2, y2 = box
        
        # Find points that project inside the 2D box
        mask = (points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) & \
               (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2)
        
        points_in_box = pointcloud[mask]
        
        # Check if we have enough points
        if len(points_in_box) < min_points:
            valid_boxes.append(False)
            boxes_3d.append(None)
            point_masks.append(mask)
            continue

        # Compute 3D box corners using min/max of points
        min_corner = points_in_box.min(axis=0)
        max_corner = points_in_box.max(axis=0)
        
        # Create 8 corners of 3D box
        corners_3d = np.array([
            [min_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]]
        ])

        boxes_3d.append(corners_3d)
        point_masks.append(mask)
        valid_boxes.append(True)

    return boxes_3d, point_masks, np.array(valid_boxes)

def get_3d_box_faces():
    """
    Returns the indices for connecting box corners into faces.
    Useful for visualization.
    
    Returns:
        list of tuples: Each tuple contains indices for one face
    """
    return [
        # Bottom face
        (0, 1, 2), (0, 2, 3),
        # Top face
        (4, 5, 6), (4, 6, 7),
        # Front face
        (0, 1, 5), (0, 5, 4),
        # Back face
        (2, 3, 7), (2, 7, 6),
        # Left face
        (0, 3, 7), (0, 7, 4),
        # Right face
        (1, 2, 6), (1, 6, 5)
    ]

def get_3d_box_edges():
    """
    Returns the indices for connecting box corners into edges.
    Useful for visualization.
    
    Returns:
        list of tuples: Each tuple contains indices for one edge
    """
    return [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Connecting edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

class Network_3D():
    def __init__(self, config):
        self.model = get_model(config["network3d"]["pretrained_path"])
        self.model.eval()
        self.device = torch.device("cuda:0")
        self.model.to(self.device)
    
    def get_class_agnostic_masks(self, pointcloud_file, datatype="point cloud", point2segment=None):
        data, raw_points, colors, features, unique_map, inverse_map, point2segment, point2segment_full = prepare_data(pointcloud_file, datatype, self.device)
        with torch.no_grad():
            # the orig code should pass in points instead of features (which is color info)
            outputs = self.model(data, raw_coordinates=raw_points, point2segment=point2segment if point2segment is not None else None)
        return map_output_to_pointcloud(outputs, inverse_map, 
                                        point2segment if not isinstance(point2segment, dict) else point2segment['superpoint'], 
                                        point2segment_full if not isinstance(point2segment_full, dict) else point2segment_full['superpoint'])        
        