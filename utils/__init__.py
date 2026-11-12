from utils.utils_3d import Network_3D
from utils.utils_2d import Network_2D, load_yaml
import time
import torch
import os
import os.path as osp
import imageio
import glob
import open3d as o3d
import numpy as np
import math
from models.Mask3D.mask3d import load_mesh_or_pc
import colorsys
from tqdm import tqdm
import pyviz3d.visualizer as viz
import logging
from datetime import datetime
import copy 

# Import evaluation utilities for ground truth loading
try:
    import sys
    sys.path.append('evaluate/scannet200')
    import util_3d
    from scannet_constants import VALID_CLASS_IDS_200 as VALID_CLASS_IDS, CLASS_LABELS_200 as CLASS_LABELS
    GT_LOADING_AVAILABLE = True
except ImportError:
    GT_LOADING_AVAILABLE = False
    print("Warning: Ground truth loading not available - evaluation modules not found")
        
def get_iou(masks):
    masks = masks.float()
    intersection = torch.einsum('ij,kj -> ik', masks, masks)
    num_masks = masks.shape[0]
    masks_batch_size = 2 # scannet 200: 20
    if masks_batch_size < num_masks:
        ratio = num_masks//masks_batch_size
        remaining = num_masks-ratio*masks_batch_size
        start_masks = list(range(0,ratio*masks_batch_size, masks_batch_size))
        if remaining == 0:
            end_masks = list(range(masks_batch_size,(ratio+1)*masks_batch_size,masks_batch_size))
        else:
            end_masks = list(range(masks_batch_size,(ratio+1)*masks_batch_size,masks_batch_size))
            end_masks[-1] = num_masks
    else:
        start_masks = [0]
        end_masks = [num_masks]
    union = torch.cat([((masks[st:ed, None, :]+masks[None, :, :]) >= 1).sum(-1) for st,ed in zip(start_masks, end_masks)])
    iou = torch.div(intersection,union)
    
    return iou

def apply_nms(masks, scores, nms_th):
    masks = masks.permute(1,0)
    scored_sorted, sorted_scores_indices = torch.sort(scores, descending=True)
    inv_sorted_scores_indices = {sorted_id.item(): id for id, sorted_id in enumerate(sorted_scores_indices)}
    maskes_sorted = masks[sorted_scores_indices]
    iou = get_iou(maskes_sorted)
    available_indices = torch.arange(len(scored_sorted))
    for indx in range(len(available_indices)):
        remove_indices = torch.where(iou[indx,indx+1:] > nms_th)[0]
        available_indices[indx+1:][remove_indices] = 0
    remaining = available_indices.unique()
    keep_indices = torch.tensor([inv_sorted_scores_indices[id.item()] for id in remaining])
    return keep_indices

def generate_vibrant_colors(num_colors):
    colors = []
    hue_increment = 1.0 / num_colors
    saturation = 1.0
    value = 1.0
    
    for i in range(num_colors):
        hue = i * hue_increment
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    
    return colors

def get_visibility_mat(pred_masks_3d, inside_mask, topk = 15):
    intersection = torch.einsum("ik, fk -> if", pred_masks_3d.float(), inside_mask.float())
    total_point_number = pred_masks_3d[:, None, :].float().sum(dim = -1)
    visibility_matrix = intersection/total_point_number
    
    if topk > visibility_matrix.shape[-1]:
        topk = visibility_matrix.shape[-1]
    
    max_visiblity_in_frame = torch.topk(visibility_matrix, topk, dim = -1).indices
    
    visibility_matrix_bool = torch.zeros_like(visibility_matrix).bool()
    visibility_matrix_bool[torch.tensor(range(len(visibility_matrix_bool)))[:, None],max_visiblity_in_frame] = True
    
    return visibility_matrix_bool

def compute_iou(box, boxes):
    assert box.shape == (4,), "Reference box must be of shape (4,)"
    assert boxes.shape[1] == 4, "Boxes must be of shape (N, 4)"
    
    x1_inter = torch.max(box[0], boxes[:, 0])
    y1_inter = torch.max(box[1], boxes[:, 1])
    x2_inter = torch.min(box[2], boxes[:, 2])
    y2_inter = torch.min(box[3], boxes[:, 3])
    inter_area = (x2_inter - x1_inter).clamp(0) * (y2_inter - y1_inter).clamp(0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    iou = inter_area / union_area
    
    return iou

class OpenYolo3D():
    def __init__(self, openyolo3d_config = ""):
        config = load_yaml(openyolo3d_config)
        self.network_3d = Network_3D(config)
        self.network_2d = Network_2D(config)
        self.openyolo3d_config = config
        
        # Set up logging
        self._setup_logging()

    def predict(self, path_2_scene_data, depth_scale, text = None, datatype="point cloud", processed_scene = None, path_to_3d_masks = None, path_to_2d_detections = None, is_gt=False, save_2d_detections=False):
        self.num_classes = len(text)+1 if text is not None else len(self.openyolo3d_config["network2d"]["text_prompts"])+1
        self.datatype = datatype
        self.world2cam = WORLD_2_CAM(path_2_scene_data, depth_scale, self.openyolo3d_config)
        self.mesh_projections = self.world2cam.get_mesh_projections()
        self.scaling_params = [self.world2cam.depth_resolution[0]/self.world2cam.image_resolution[0], self.world2cam.depth_resolution[1]/self.world2cam.image_resolution[1]]
        scene_name = path_2_scene_data.split("/")[-1]
        
        # Set up file logging now that we have the scene name
        self._setup_file_logging(scene_name)
        
        self._log("[🚀 ACTION] 3D mask proposals computation ...")
        start = time.time()
        # get 3D instance masks
        if path_to_3d_masks is None:
            self.preds_3d = self.network_3d.get_class_agnostic_masks(self.world2cam.mesh, datatype) if processed_scene is None else self.network_3d.get_class_agnostic_masks(processed_scene, datatype)
            keep_score = self.preds_3d[1] >= self.openyolo3d_config["network3d"]["th"]
            keep_score = keep_score.cpu() 
            keep_nms = apply_nms(self.preds_3d[0][:, keep_score].cuda(), self.preds_3d[1][keep_score].cuda(), self.openyolo3d_config["network3d"]["nms"])
            self.preds_3d = (self.preds_3d[0].cpu().permute(1,0)[keep_score][keep_nms].permute(1,0), self.preds_3d[1].cpu()[keep_score][keep_nms])
            # torch.save(self.preds_3d, save_path)
        else:
            try:
                self.preds_3d = torch.load(osp.join(path_to_3d_masks, f"{scene_name}.pt"))
            except FileNotFoundError:
                msk = torch.load(osp.join(path_to_3d_masks, f"{scene_name}_masks.pt"))
                msk = torch.from_numpy(msk).to(torch.float32)
                conf = torch.ones(msk.shape[1])
                self.preds_3d = (msk, conf)
            
        self._log(f"[🕒 INFO] Elapsed time {(time.time()-start)}")
        self._log(f"[✅ INFO] Proposals computed.")   
        self._log("[🚀 ACTION] 2D Bounding Boxes computation ...")
        start = time.time()

        # get 2D bounding boxes
        if path_to_2d_detections is None:
            # Compute 2D detections and optionally save them
            dataset = 'scannet200' if 'scannet200' in path_2_scene_data else 'replica'
            save_path = osp.join("output", dataset, "2d_detections", f"{scene_name}.pt") if save_2d_detections else None
            self.preds_2d = self.network_2d.get_bounding_boxes(self.world2cam.color_paths, text, save_path=save_path)
        else:
            # Load cached 2D detections
            try:
                path_2d = osp.join(path_to_2d_detections, f"{scene_name}.pt")
                detection_data = torch.load(path_2d)
                self.preds_2d = detection_data['detections']
                self._log(f"[✅ INFO] Loaded cached 2D detections from {path_2d}")
                
                # Validate that text prompts match (if specified)
                if text is not None:
                    cached_prompts = detection_data.get('text_prompts', [])
                    if set(text) != set(cached_prompts):
                        self._log(f"[⚠️ WARNING] Text prompts mismatch. Cached: {cached_prompts}, Requested: {text}", "WARNING")
                
            except FileNotFoundError:
                self._log(f"[⚠️ WARNING] Cached detections not found at {path_to_2d_detections}, computing fresh...", "WARNING")
                save_path = osp.join(path_to_2d_detections, f"{scene_name}.pt")
                self.preds_2d = self.network_2d.get_bounding_boxes(self.world2cam.color_paths, text, save_path=save_path)
        
        self._log(f"[🕒 INFO] Elapsed time {(time.time()-start)}")
        self._log(f"[✅ INFO] Bounding boxes ready.")  

        # # Store 2D-to-3D projection analysis for debugging
            
        # Filter projected boxes against existing 3D masks 
        box_filtering_enabled = self.openyolo3d_config.get('box_filtering', {}).get('enabled', True)
        load_novel_masks = self.openyolo3d_config.get('box_filtering', {}).get('load_novel_masks', True)

        if box_filtering_enabled:          
            # Get scene name from processed_scene path
            pcd_file = processed_scene if processed_scene is not None else self.world2cam.mesh
            scene_name = osp.basename(osp.dirname(pcd_file))
            
            # Check if cached merged candidates exist
            dataset = 'scannet200' if 'scannet200' in pcd_file else 'replica'
            cache_dir = osp.join('output', dataset, '3d_merged_candidates_rerun')
            cache_path = osp.join(cache_dir, f"{scene_name}.pt")
            
            if osp.exists(cache_path) and load_novel_masks:
                self._log(f"[📁 INFO] Loading cached merged candidates from {cache_path}")
                try:
                    self.merged_novel_candidates = torch.load(cache_path)
                    self._log(f"[✅ INFO] Loaded {len(self.merged_novel_candidates)} cached novel candidates")
                except Exception as e:
                    self._log(f"[⚠️ WARNING] Failed to load cached candidates: {e}", "WARNING")
                    self._log("[🚀 ACTION] Computing candidates from scratch...")
                    self.projected_3d_boxes = self._compute_2dto3d_projection()            
                    self.merged_novel_candidates = self._compute_novel_candidates(pcd_file, cache_path)
            else:
                self._log(f"[🚀 ACTION] No cached candidates found, computing from scratch...")
                self.projected_3d_boxes = self._compute_2dto3d_projection()            
                self.merged_novel_candidates = self._compute_novel_candidates(pcd_file, cache_path)

            # Convert novel candidates to binary masks and merge with 3D network masks
            if len(self.merged_novel_candidates) > 0:
                self._log("[🚀 ACTION] Integrating novel candidates with 3D masks...")
                start = time.time()
                self._integrate_novel_candidates_with_3d_masks()
                self._log(f"[🕒 INFO] Integration elapsed time {(time.time()-start)}")
                
        else:
            self._log("[⚠️ INFO] Box filtering disabled in config, skipping filtering and merging")
            self.filtered_projected_boxes = {}
            self.merged_novel_candidates = []
            
        self._log("[🚀 ACTION] Predicting all proposal masks...")
        out_path = osp.join("output/scannet200/scannet200_our_masks", scene_name + '.pt')
        start = time.time()
        prediction = self.label_3d_masks_from_2d_bboxes(scene_name, is_gt)
        self._log(f"[🕒 INFO] Elapsed time {(time.time()-start)}")
        
        # Store additional debug information
        self.final_prediction = prediction
                                
        return prediction
    
    def _compute_2dto3d_projection(self):
        """
        Compute 2D-to-3D projection analysis for all frames once
        This avoids redundant computation in debug visualization
        """
        self._log("[🔍 DEBUG] Computing projection analysis...", "DEBUG")
        projection_data = {}
        count = 0
        dataset, scene_id = osp.dirname(self.world2cam.mesh).split('/')[-2:]
        data_dir = osp.join('output', dataset, '3d_projected', scene_id)

        detection_res = copy.deepcopy(self.preds_2d)
        for frame_id, detections in tqdm(detection_res.items(), desc="Projecting 2D boxes to 3D"):
            # if count > 1:
            #     break 
            # count += 1
            data_path = osp.join(data_dir, frame_id + ".pt")                            
            try:
                projection_data[frame_id] = torch.load(data_path)
            except:
                # Compute 3D projections for this frame (expensive operation done once)
                projection_data[frame_id] = self.world2cam.get_3d_bboxes_from_frame(
                    frame_id, detections
                ) 
            # projection_data[frame_id] = self.world2cam.get_3d_bboxes_from_frame(
            #     frame_id, detections
            # )                 
                
        self._log(f"[✅ DEBUG] Projection analysis completed for {len(projection_data)} frames", "DEBUG")
        return projection_data
    
    def visualize_projected_boxes(self, processed_scene, output_dir="visualization_3d", show_ground_truth=True):
        """
        Create interactive visualization of point cloud and projected 3D boxes using pyviz3d.
        The UI allows users to toggle visibility of different elements:
        - Whole scene point cloud
        - Individual 3D boxes with their projected points from each frame
        - 3D bounding box wireframes
        - Ground truth instance masks (if show_ground_truth=True)

        Args:
            processed_scene (str): Path to the .npy file containing all scene points.
            output_dir (str): Directory to save the interactive visualization.
            show_ground_truth (bool): Whether to include ground truth instances in visualization.
        
        Note: Uses self.projected_3d_boxes from _compute_projection_analysis()
        """
        # Create output directory
        scene_name = osp.dirname(processed_scene).split("/")[-1]
        os.makedirs(osp.join(output_dir, scene_name), exist_ok=True)
        
        # Load the processed scene data
        if isinstance(processed_scene, str):
            scene_points = np.load(processed_scene)
        else:
            scene_points = processed_scene
            
        # Handle scene points, colors, and segments
        segments = None  # Initialize segments variable

        if scene_points.shape[1] >= 12:
            # Full format: [x, y, z, r, g, b, normal_x, normal_y, normal_z, segment_id, label1, label2]
            points, colors, normals, segments, labels = (
                scene_points[:, :3],      # xyz coordinates
                scene_points[:, 3:6],     # rgb colors
                scene_points[:, 6:9],     # normals
                scene_points[:, 9],       # segment IDs
                scene_points[:, 10:12],   # labels
            )
            scene_coords = points
            scene_colors = colors
            print(f"[📊 INFO] Parsed full format with {len(np.unique(segments))} unique segments")
        elif scene_points.shape[1] >= 6:
            # Legacy format: [x, y, z, r, g, b]
            scene_coords = scene_points[:, :3]  # xyz coordinates
            scene_colors = scene_points[:, 3:6]  # rgb colors
            print(f"[📊 INFO] Parsed legacy format [x, y, z, r, g, b]")
        elif scene_points.shape[1] > 3:
            # Intermediate format
            scene_coords = scene_points[:, :3]  # xyz coordinates
            scene_colors = np.full((len(scene_coords), 3), [0.7, 0.7, 0.7])  # Default gray
            print(f"[📊 INFO] Parsed intermediate format with {scene_points.shape[1]} columns")
        else:
            # Minimal format: [x, y, z]
            scene_coords = scene_points
            scene_colors = np.full((len(scene_coords), 3), [0.7, 0.7, 0.7])  # Default gray
            print(f"[📊 INFO] Parsed minimal format [x, y, z]")
        
        # Normalize coordinates: center the scene at origin for easier rotation
        scene_center = np.mean(scene_coords, axis=0)
        scene_coords_normalized = scene_coords - scene_center
        print(f"[🔄 INFO] Scene centered at origin. Original center: {scene_center}")
        
        # Initialize the visualizer
        v = viz.Visualizer()
        
        # Get projection data from the stored analysis
        if not hasattr(self, 'projected_3d_boxes'):
            raise ValueError("No projection analysis found. Run predict() first to compute projections.")
        
        # Generate colors for visualization (no longer needed for boxes)
        # colors = self._generate_vibrant_colors_for_viz(total_boxes)

        # Process superpoint segments if available
        if segments is not None:
            unique_segments = np.unique(segments)
            n_segments = len(unique_segments)
            print(f"[🔍 INFO] Found {n_segments} unique superpoint segments")

            # Generate distinct colors for segments
            segment_colors = self._generate_distinct_colors(n_segments)

            # Create color mapping for each point based on its segment
            all_segment_points = []
            all_segment_colors = []

            for i, segment_id in enumerate(unique_segments):
                segment_mask = (segments == segment_id)
                segment_points = scene_coords_normalized[segment_mask]

                if len(segment_points) == 0:
                    continue

                # Assign unique color to this segment
                segment_color = segment_colors[i % len(segment_colors)]
                point_colors = np.full((len(segment_points), 3), segment_color)

                # Collect all points and colors for single toggle
                all_segment_points.extend(segment_points)
                all_segment_colors.extend(point_colors)

            # Add single toggle for all segments with distinct colors
            if all_segment_points:
                v.add_points(
                    name=f"Superpoint_Segments_{n_segments}_segments",
                    positions=np.array(all_segment_points),
                    colors=np.array(all_segment_colors) * 255,
                    point_size=20.0,
                    visible=False  # Hidden by default
                )
        else:
            print("[⚠️ INFO] No segment information available in scene data")
            n_segments = 0
        
        # Add whole scene point cloud with actual colors (toggleable)
        # Ensure colors are in 0-255 range for pyviz3d
        if scene_colors.max() <= 1.0:
            # Colors are in 0-1 range, convert to 0-255
            scene_colors_viz = scene_colors * 255
        else:
            # Colors are already in 0-255 range
            scene_colors_viz = scene_colors
            
        v.add_points(
            name="Scene_PointCloud", 
            positions=scene_coords_normalized,
            colors=scene_colors_viz,
            point_size=20.0,
            visible=True
        )
        
        # Add ground truth instances if requested
        if show_ground_truth:
            self._log("[🎨 INFO] Loading ground truth instances for visualization...", "INFO")
            gt_instances = self._load_ground_truth_instances(processed_scene)

            if gt_instances:
                # Import category definitions
                try:
                    from evaluate.scannet200.scannet_constants import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
                    self._log("[✅ INFO] Loaded category definitions from scannet_constants", "INFO")
                except ImportError:
                    self._log("[❌ ERROR] Cannot import category definitions, using default grouping", "ERROR")
                    HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200 = set(), set(), set()

                # Group instances by category
                instances_by_category = {'HEAD': [], 'COMMON': [], 'TAIL': []}

                # Generate distinct colors for all GT instances
                gt_colors = self._generate_distinct_colors(len(gt_instances))

                self._log(f"[🎨 INFO] Adding {len(gt_instances)} ground truth instances to visualization", "INFO")

                # Process each instance and categorize
                for instance_idx, (gt_id, instance_data) in enumerate(gt_instances.items()):
                    class_label = instance_data['class_label']
                    instance_id = instance_data['instance_id']
                    point_count = instance_data['point_count']

                    # Determine category
                    if class_label in HEAD_CATS_SCANNET_200:
                        category = 'HEAD'
                    elif class_label in COMMON_CATS_SCANNET_200:
                        category = 'COMMON'
                    elif class_label in TAIL_CATS_SCANNET_200:
                        category = 'TAIL'
                    else:
                        category = 'HEAD'  # Default to HEAD for unknown classes

                    # Get instance points and normalize coordinates
                    instance_points = instance_data['points']
                    instance_points_normalized = instance_points - scene_center

                    # Apply unique color to this instance
                    instance_color = gt_colors[instance_idx % len(gt_colors)]

                    # Add to category group
                    instances_by_category[category].append({
                        'points': instance_points_normalized,
                        'color': instance_color,
                        'gt_id': gt_id,
                        'class_label': class_label,
                        'instance_id': instance_id,
                        'point_count': point_count
                    })

                    # Skip individual instance toggle to avoid menu clutter

                # Add category groups (3 toggles for ground truth)
                for category, instances in instances_by_category.items():
                    if not instances:
                        continue

                    # Collect all points and colors for this category
                    category_points = []
                    category_colors = []

                    for instance in instances:
                        category_points.extend(instance['points'])
                        point_colors = np.full((len(instance['points']), 3), instance['color'])
                        category_colors.extend(point_colors)

                    if category_points:
                        v.add_points(
                            name=f"GroundTruth_{category}_{len(instances)}_objects",
                            positions=np.array(category_points),
                            colors=np.array(category_colors) * 255,
                            point_size=25.0,
                            visible=False  # Hidden by default, can be toggled by category
                        )

                self._log(f"[✅ INFO] Ground truth by categories - HEAD: {len(instances_by_category['HEAD'])}, "
                         f"COMMON: {len(instances_by_category['COMMON'])}, TAIL: {len(instances_by_category['TAIL'])}", "INFO")

                # Group COMMON and TAIL instances by specific class labels for additional toggles
                common_classes = {}
                tail_classes = {}

                # Group COMMON instances by class label
                for instance in instances_by_category['COMMON']:
                    class_label = instance['class_label']
                    if class_label not in common_classes:
                        common_classes[class_label] = []
                    common_classes[class_label].append(instance)

                # Group TAIL instances by class label
                for instance in instances_by_category['TAIL']:
                    class_label = instance['class_label']
                    if class_label not in tail_classes:
                        tail_classes[class_label] = []
                    tail_classes[class_label].append(instance)

                self._log(f"[🔍 INFO] Class-level breakdown - COMMON: {len(common_classes)} classes, TAIL: {len(tail_classes)} classes", "INFO")

                # Add class-level toggles for COMMON objects
                for class_label, class_instances in common_classes.items():
                    if not class_instances:
                        continue

                    # Collect all points and colors for this COMMON class (preserving individual instance colors)
                    class_points = []
                    class_colors = []

                    for instance in class_instances:
                        class_points.extend(instance['points'])
                        point_colors = np.full((len(instance['points']), 3), instance['color'])  # Use original instance color
                        class_colors.extend(point_colors)

                    if class_points:
                        v.add_points(
                            name=f"GroundTruth_COMMON_{class_label}_{len(class_instances)}_instances",
                            positions=np.array(class_points),
                            colors=np.array(class_colors) * 255,
                            point_size=25.0,
                            visible=False  # Hidden by default
                        )

                # Add class-level toggles for TAIL objects
                for class_label, class_instances in tail_classes.items():
                    if not class_instances:
                        continue

                    # Collect all points and colors for this TAIL class (preserving individual instance colors)
                    class_points = []
                    class_colors = []

                    for instance in class_instances:
                        class_points.extend(instance['points'])
                        point_colors = np.full((len(instance['points']), 3), instance['color'])  # Use original instance color
                        class_colors.extend(point_colors)

                    if class_points:
                        v.add_points(
                            name=f"GroundTruth_TAIL_{class_label}_{len(class_instances)}_instances",
                            positions=np.array(class_points),
                            colors=np.array(class_colors) * 255,
                            point_size=25.0,
                            visible=False  # Hidden by default
                        )

                # Note: Label statistics will be written after visualization save to avoid deletion
            else:
                self._log("[⚠️ WARNING] No ground truth instances found for visualization", "WARNING")
                instances_by_category = {'HEAD': [], 'COMMON': [], 'TAIL': []}
                common_classes = {}
                tail_classes = {}

        # Add 3D backbone masks (1 toggle for 3D predictions)
        if hasattr(self, 'original_3d_masks'):
            self._log("[🎨 INFO] Adding 3D backbone masks...", "INFO")
            original_masks, original_scores = self.original_3d_masks  # [N_points, N_original], [N_original]
            n_original = original_masks.shape[1]

            # Generate distinct colors for 3D masks
            colors_3d = self._generate_distinct_colors(n_original)

            all_3d_points = []
            all_3d_colors = []

            for mask_idx in range(n_original):
                mask = original_masks[:, mask_idx].bool()
                score = original_scores[mask_idx].item()

                if mask.sum() == 0:
                    continue

                # Get mask points
                mask_points = scene_coords_normalized[mask]
                mask_color = colors_3d[mask_idx % len(colors_3d)]
                point_colors = np.full((len(mask_points), 3), mask_color)

                # Collect for group visualization (no individual toggles)
                all_3d_points.extend(mask_points)
                all_3d_colors.extend(point_colors)

            # Add all 3D masks group (1 toggle for 3D predictions)
            if all_3d_points:
                v.add_points(
                    name=f"Prediction_3D_Backbone_{n_original}_masks",
                    positions=np.array(all_3d_points),
                    colors=np.array(all_3d_colors) * 255,
                    point_size=25.0,
                    visible=False  # Hidden by default
                )

            self._log(f"[✅ INFO] Added {n_original} 3D backbone masks", "INFO")

        # Add novel RGB-D masks (1 toggle for novel predictions)
        if hasattr(self, 'novel_3d_masks'):
            self._log("[🎨 INFO] Adding novel RGB-D masks...", "INFO")
            novel_masks, novel_scores = self.novel_3d_masks  # [N_points, N_novel], [N_novel]
            n_novel = novel_masks.shape[1]

            # Generate distinct colors for novel masks
            colors_novel = self._generate_distinct_colors(n_novel)

            all_novel_points = []
            all_novel_colors = []

            for mask_idx in range(n_novel):
                mask = novel_masks[:, mask_idx].bool()
                score = novel_scores[mask_idx].item()

                if mask.sum() == 0:
                    continue

                # Get mask points
                mask_points = scene_coords_normalized[mask]
                mask_color = colors_novel[mask_idx % len(colors_novel)]
                point_colors = np.full((len(mask_points), 3), mask_color)

                # Collect for group visualization (no individual toggles)
                all_novel_points.extend(mask_points)
                all_novel_colors.extend(point_colors)

            # Add all novel masks group (1 toggle for novel predictions)
            if all_novel_points:
                v.add_points(
                    name=f"Prediction_Novel_RGBD_{n_novel}_masks",
                    positions=np.array(all_novel_points),
                    colors=np.array(all_novel_colors) * 255,
                    point_size=27.0,
                    visible=False  # Hidden by default
                )

            self._log(f"[✅ INFO] Added {n_novel} novel RGB-D masks", "INFO")
                    
        # Box visualization removed - focusing on instance masks only
        
        # Add metadata information
        self._add_visualization_metadata(v, len(scene_coords_normalized))
        
        # Save interactive visualization
        output_path = os.path.join(output_dir, scene_name)
        v.save(output_path)

        # Write label statistics AFTER visualization save to avoid being deleted by pyviz3d
        if show_ground_truth and 'instances_by_category' in locals():
            self._write_label_statistics(instances_by_category, output_dir, scene_name)
        
        print(f"[✅ INFO] Interactive visualization saved to: {output_path}/index.html")
        print(f"[📋 INFO] Visualization contains:")
        print(f"  - Scene point cloud: {len(scene_coords_normalized):,} points with colors (visible by default)")
        if segments is not None:
            print(f"  - Superpoint segments: {n_segments} segments with distinct colors (1 toggle)")
        if show_ground_truth:
            # Calculate toggle counts
            total_gt_toggles = 3  # HEAD + COMMON + TAIL category toggles
            class_toggles = 0
            if 'common_classes' in locals():
                class_toggles += len(common_classes)
            if 'tail_classes' in locals():
                class_toggles += len(tail_classes)
            total_gt_toggles += class_toggles

            print(f"  - Ground truth masks:")
            print(f"    * Category-level: HEAD (1 toggle), COMMON (1 toggle), TAIL (1 toggle)")
            if 'common_classes' in locals() and common_classes:
                print(f"    * COMMON class-level: {len(common_classes)} class toggles ({', '.join(sorted(common_classes.keys()))})")
            if 'tail_classes' in locals() and tail_classes:
                print(f"    * TAIL class-level: {len(tail_classes)} class toggles ({', '.join(sorted(tail_classes.keys()))})")
            print(f"    * Total ground truth toggles: {total_gt_toggles}")
        if hasattr(self, 'original_3d_masks'):
            print(f"  - 3D backbone prediction masks (1 toggle)")
        if hasattr(self, 'novel_3d_masks'):
            print(f"  - Novel RGB-D prediction masks (1 toggle)")
        print(f"  - All coordinates normalized for easier rotation")
        if segments is not None:
            print(f"  - Toggle between original colors (Scene_PointCloud) and segment colors (Superpoint_Segments)")
        print(f"  - Use sidebar toggles to show/hide different mask categories and individual classes")
        print(f"[🌐 INFO] Open {output_path}/index.html in a web browser to view")

    def _generate_vibrant_colors_for_viz(self, num_colors):
        """Generate vibrant colors for visualization"""
        colors = []
        for i in range(num_colors):
            hue = i / max(num_colors, 1)
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors
    
    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert a 3x3 rotation matrix to quaternion (w, x, y, z).
        """
        # Ensure it's a numpy array
        R = np.array(R)
        
        # Calculate quaternion components
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        # Return quaternion as [w, x, y, z]
        return np.array([w, x, y, z])
    
    
    def _add_visualization_metadata(self, visualizer, num_scene_points):
        """Add metadata display to visualization"""
        # Create summary info that will be displayed
        metadata_text = [
            f"Scene Points: {num_scene_points:,}",
            f"Dataset: {getattr(self, 'datatype', 'Unknown')}",
            "Toggle mask categories using the sidebar controls"
        ]
        
        # pyviz3d doesn't have direct text support, but we can create a small info point
        info_point = np.array([[0, 0, 0]])  # Origin point for reference
        info_color = np.array([[1.0, 1.0, 1.0]])  # White
        
        visualizer.add_points(
            name="Info_Origin_Reference",
            positions=info_point,
            colors=info_color * 255,
            point_size=50.0,
            visible=True
        )

    def _write_label_statistics(self, instances_by_category, output_dir, scene_name):
        """Write detailed label statistics file for the scene."""
        # Ensure the output directory exists
        scene_output_dir = os.path.join(output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        stats_file = os.path.join(scene_output_dir, "label_statistics.txt")

        print(f"[🔄 DEBUG] Attempting to write statistics to: {stats_file}")
        print(f"[🔄 DEBUG] Directory exists: {os.path.exists(scene_output_dir)}")

        try:
            # Import category definitions for detailed class listings
            from evaluate.scannet200.scannet_constants import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
        except ImportError:
            HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200 = set(), set(), set()

        try:
            with open(stats_file, 'w') as f:
                f.write(f"Label Statistics for Scene: {scene_name}\n")
                f.write("=" * 50 + "\n\n")

                # Summary counts
                total_instances = sum(len(instances) for instances in instances_by_category.values())
                f.write(f"Total instances: {total_instances}\n")
                f.write(f"HEAD objects: {len(instances_by_category['HEAD'])}\n")
                f.write(f"COMMON objects: {len(instances_by_category['COMMON'])}\n")
                f.write(f"TAIL objects: {len(instances_by_category['TAIL'])}\n\n")

                # Detailed breakdown by category
                for category in ['HEAD', 'COMMON', 'TAIL']:
                    instances = instances_by_category[category]
                    if not instances:
                        f.write(f"{category} Category: 0 objects\n\n")
                        continue

                    f.write(f"{category} Category: {len(instances)} objects\n")
                    f.write("-" * 30 + "\n")

                    # Group instances by class label
                    class_counts = {}
                    for instance in instances:
                        class_label = instance['class_label']
                        if class_label not in class_counts:
                            class_counts[class_label] = []
                        class_counts[class_label].append(instance)

                    # Sort classes by count (descending)
                    sorted_classes = sorted(class_counts.items(), key=lambda x: len(x[1]), reverse=True)

                    for class_label, class_instances in sorted_classes:
                        f.write(f"  {class_label}: {len(class_instances)} instance(s)\n")
                        for instance in class_instances:
                            f.write(f"    - Instance ID {instance['instance_id']}: {instance['point_count']} points\n")

                    f.write("\n")

                # Available classes in each category (reference)
                f.write("Category Definitions (Reference)\n")
                f.write("=" * 30 + "\n")
                f.write(f"HEAD categories ({len(HEAD_CATS_SCANNET_200)}): {', '.join(sorted(HEAD_CATS_SCANNET_200))}\n\n")
                f.write(f"COMMON categories ({len(COMMON_CATS_SCANNET_200)}): {', '.join(sorted(COMMON_CATS_SCANNET_200))}\n\n")
                f.write(f"TAIL categories ({len(TAIL_CATS_SCANNET_200)}): {', '.join(sorted(TAIL_CATS_SCANNET_200))}\n")

            # Verify file was written successfully
            if os.path.exists(stats_file):
                file_size = os.path.getsize(stats_file)
                print(f"[✅ SUCCESS] Label statistics file created: {stats_file} ({file_size} bytes)")
                self._log(f"[📊 INFO] Label statistics saved to: {stats_file} ({file_size} bytes)", "INFO")
            else:
                print(f"[❌ ERROR] Label statistics file was not created: {stats_file}")
                self._log(f"[❌ ERROR] Label statistics file was not created: {stats_file}", "ERROR")

        except Exception as e:
            error_msg = f"Error writing label statistics: {str(e)}"
            print(f"[❌ ERROR] {error_msg}")
            self._log(f"[❌ ERROR] {error_msg}", "ERROR")
            raise

    def label_3d_masks_from_2d_bboxes(self, scene_name, is_gt=False):
        projections_mesh_to_frame , keep_visible_points = self.mesh_projections
        predictions_2d_bboxes = self.preds_2d
        prediction_3d_masks, _ = self.preds_3d
        predicted_masks, predicated_classes, predicated_scores = self.label_3d_masks_from_label_maps(prediction_3d_masks.bool(), 
                                                                                                        predictions_2d_bboxes, 
                                                                                                        projections_mesh_to_frame,
                                                                                                        keep_visible_points, 
                                                                                                        is_gt)
        
        self.predicted_masks = predicted_masks
        self.predicated_scores = predicated_scores
        self.predicated_classes = predicated_classes
        
        return {scene_name : (predicted_masks, predicated_classes, predicated_scores)}
    
    
    def label_3d_masks_from_label_maps(self, 
                                        prediction_3d_masks, 
                                        predictions_2d_bboxes, 
                                        projections_mesh_to_frame, 
                                        keep_visible_points,
                                        is_gt):
        
        label_maps = self.construct_label_maps(predictions_2d_bboxes) #construct the label maps , start from the biggest bbox to small one

        visibility_matrix = get_visibility_mat(prediction_3d_masks.cuda().permute(1,0), keep_visible_points.cuda(), topk = 25 if is_gt else self.openyolo3d_config["openyolo3d"]["topk"])
        valid_frames = visibility_matrix.sum(dim=0) >= 1
        
        prediction_3d_masks = prediction_3d_masks.permute(1,0).cpu()
        prediction_3d_masks_np = prediction_3d_masks.numpy()
        valid_frames = valid_frames.to(projections_mesh_to_frame.device)
        projections_mesh_to_frame = projections_mesh_to_frame[valid_frames].cpu().numpy()
        visibility_matrix = visibility_matrix[:, valid_frames].cpu().numpy()
        keep_visible_points = keep_visible_points[valid_frames].cpu().numpy()
        distributions = []
        
        class_labels = []
        class_probs = []
        class_dists = []
        label_maps = label_maps[valid_frames].numpy()
        bounding_boxes = predictions_2d_bboxes.values()
        bounding_boxes_valid = [bbox for (bi, bbox) in enumerate(bounding_boxes) if valid_frames[bi]]
        for mask_id, mask in enumerate(prediction_3d_masks_np):
            prob_normalizer = 0

            representitive_frame_ids = np.where(visibility_matrix[mask_id])[0]
            labels_distribution = []
            iou_vals = []
            for representitive_frame_id in representitive_frame_ids:
                visible_points_mask = (keep_visible_points[representitive_frame_id].squeeze()*mask).astype(bool)
                prob_normalizer +=  visible_points_mask.sum()
                instance_x_y_coords = projections_mesh_to_frame[representitive_frame_id][np.where(visible_points_mask)].astype(np.int64)
                
                boxes = bounding_boxes_valid[representitive_frame_id]["bbox"].long()
                if len(boxes) > 0 and len(instance_x_y_coords) > 10:
                    x_l, x_r, y_t, y_b = instance_x_y_coords[:, 0].min(), instance_x_y_coords[:, 0].max()+1, instance_x_y_coords[:, 1].min(), instance_x_y_coords[:, 1].max()+1
                    box = torch.tensor([x_l/self.scaling_params[1], y_t/self.scaling_params[0], x_r/self.scaling_params[1], y_b/self.scaling_params[0]])
                
                    iou_values = compute_iou(box, boxes)
                    iou_vals.append(iou_values.max().item())
                selected_labels = label_maps[representitive_frame_id, instance_x_y_coords[:, 1], instance_x_y_coords[:, 0]]
                labels_distribution.append(selected_labels)
            
            labels_distribution = np.concatenate(labels_distribution) if len(labels_distribution) > 0 else np.array([-1])
            
            # class_dists.append(labels_distribution)
            distribution = torch.zeros(self.num_classes) if self.openyolo3d_config["openyolo3d"]["topk_per_image"] != -1 else None
            if (labels_distribution != -1).sum() != 0:
                
                if distribution is not None:
                    all_labels = torch.from_numpy(labels_distribution[labels_distribution != -1])
                    all_labels_unique = all_labels.unique()
                    for lb in all_labels_unique:
                        distribution[lb] = (all_labels == lb).sum()
                        
                    distribution = distribution/distribution.max()
                
                class_label = torch.mode(torch.from_numpy(labels_distribution[labels_distribution != -1])).values.item()
                class_prob = (labels_distribution == class_label).sum()/prob_normalizer
            else:
                if distribution is not None:
                    distribution[-1] = 1.0
                class_label = self.num_classes-1
                class_prob = 0.0

            iou_vals = torch.tensor(iou_vals)
            
            class_labels.append(class_label)
            if (iou_vals != 0).sum():
                iou_prob = iou_vals[iou_vals != 0].mean().item()
            else:
                iou_prob = 0.0
            
            class_probs.append(class_prob*iou_prob)
            if distribution is not None:
                distributions.append(distribution)
                
        pred_classes = torch.tensor(class_labels)
        pred_scores = torch.tensor(class_probs)
        if distribution is not None:
            distributions = torch.stack(distributions) if len(distributions) > 0 else torch.tensor((0, self.num_classes))
        
        if (self.openyolo3d_config["openyolo3d"]["topk_per_image"] != -1) and (not is_gt):
            # print("TOPK USED")
            n_instance = distributions.shape[0]
            distributions = distributions.reshape(-1)
            labels = (
            torch.arange(self.num_classes, device=distributions.device)
            .unsqueeze(0)
            .repeat(n_instance, 1)
            .flatten(0, 1)
            )

            cur_topk = self.openyolo3d_config["openyolo3d"]["topk_per_image"]
            _, idx = torch.topk(distributions, k=min(cur_topk, len(distributions)), largest=True)
            mask_idx = torch.div(idx, self.num_classes, rounding_mode="floor")

            pred_classes = labels[idx]
            pred_scores = distributions[idx].cuda()
            prediction_3d_masks = prediction_3d_masks[mask_idx]
        
        return prediction_3d_masks.permute(1,0), pred_classes, pred_scores
    
    def construct_label_maps(self, predictions_2d_bboxes, save_label_map=False):
        label_maps = (torch.ones((len(predictions_2d_bboxes), self.world2cam.height, self.world2cam.width))*-1).type(torch.int16)
        for frame_id, pred in enumerate(predictions_2d_bboxes.values()):
            bboxes = pred["bbox"].long()
            labels = pred["labels"].type(torch.int16)
        
            bboxes[:,0] = bboxes[:,0]*self.scaling_params[1]
            bboxes[:,2] = bboxes[:,2]*self.scaling_params[1]
            bboxes[:,1] = bboxes[:,1]*self.scaling_params[0]
            bboxes[:,3] = bboxes[:,3]*self.scaling_params[0]
            bboxes_weights = (bboxes[:,2]-bboxes[:,0])+(bboxes[:,3]-bboxes[:,1])
            sorted_indices = bboxes_weights.sort(descending=True).indices
            bboxes = bboxes[sorted_indices]
            labels = labels[sorted_indices]
            for id, bbox in enumerate(bboxes):
                label_maps[frame_id, bbox[1]:bbox[3],bbox[0]:bbox[2]] = labels[id]
                
        return label_maps
    
    def save_output_as_ply(self, save_path, th = 0.1):
        num_classes = len(self.predicated_classes.unique())
        data = load_mesh_or_pc(self.world2cam.mesh, self.datatype)
        if self.datatype == 'mesh':
            mesh = data
            vertex_colors = np.asarray(mesh.vertex_colors)
            vibrant_colors = generate_vibrant_colors(num_classes)

            for i, class_id in enumerate((self.predicated_classes).unique()):
                if class_id == self.num_classes-1:
                    continue
                class_id_mask = self.predicated_classes == class_id
                scores_per_class = self.predicated_scores[class_id_mask]
                class_id_max = torch.argmax(scores_per_class)
                mask = self.predicted_masks.permute(1,0)[class_id_mask][class_id_max]
                vertex_colors[mask] = np.array(vibrant_colors.pop())
                
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            o3d.io.write_triangle_mesh(save_path, mesh)
        elif self.datatype == 'point cloud':
            point_cloud = data
            point_colors = np.asarray(point_cloud.colors)
            vibrant_colors = generate_vibrant_colors(num_classes)
            for i, class_id in enumerate(self.predicated_classes.unique()):
                if class_id == self.num_classes-1:
                    continue
                class_id_mask = self.predicated_classes == class_id
                scores_per_class = self.predicated_scores[class_id_mask]
                class_id_max = torch.argmax(scores_per_class)
                if scores_per_class[class_id_max] < th:
                    continue
                mask = self.predicted_masks.permute(1,0)[class_id_mask][class_id_max]
                point_colors[mask] = np.array(vibrant_colors.pop())

            point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
            o3d.io.write_point_cloud(save_path, point_cloud)
        else:
            print("[ERROR] output is not saved, please check input 3D scene folder.")
            exit()
    
    # ======== Novel Object Detection Methods ========
    
    def _filter_projected_boxes_with_3d_masks(self, processed_scene):
        """
        Filter projected 3D boxes against existing 3D network masks using containment ratio.
        Remove boxes where a significant portion of existing 3D masks are contained within the box.
        
        Args:
            processed_scene (str): Path to .npy file containing scene points and superpoint assignments
            
        Returns:
            dict: Filtered boxes per frame with superpoint masks
        """
        # Load scene data with superpoints
        if processed_scene.endswith('.npy'):
            scene_data = np.load(processed_scene)
            points = torch.from_numpy(scene_data[:, :3].copy()).float()
            spp = scene_data[:, 9] if scene_data.shape[1] > 9 else scene_data[:, 6]
            superpoints = torch.from_numpy(spp.copy()).long() if spp is not None else None
        else:
            pcd = o3d.io.read_point_cloud(processed_scene)
            points = torch.from_numpy(np.asarray(pcd.points)).float()
            
        masks_3d, _ = self.preds_3d  # [N_points, N_masks] from 3D network - each column is a mask
        filtered_boxes = {}
        
        # Configuration parameters
        box_filtering_config = self.openyolo3d_config.get('box_filtering', {})
        containment_threshold = box_filtering_config.get('containment_threshold', 0.7)
        frame_sampling = box_filtering_config.get('frame_sampling', 1)
        
        # Apply frame sampling - filter only selected frames
        all_frame_ids = sorted(self.projected_3d_boxes.keys())
        if frame_sampling > 1:
            sampled_frame_ids = all_frame_ids[::frame_sampling]
            self._log(f"[🔍 DEBUG] Frame sampling for filtering: using {len(sampled_frame_ids)}/{len(all_frame_ids)} frames (every {frame_sampling} frames)", "DEBUG")
        else:
            sampled_frame_ids = all_frame_ids
            self._log(f"[🔍 DEBUG] Filtering all {len(sampled_frame_ids)} frames", "DEBUG")
        
        total_original = 0
        total_filtered = 0
        
        for frame_id in tqdm(sampled_frame_ids, desc="Filtering frames"):
            frame_boxes = self.projected_3d_boxes[frame_id]
            filtered_frame_boxes = []
            total_original += len(frame_boxes)
            
            for box in frame_boxes:
                # Check if this box is redundant with any existing 3D mask
                is_novel = True
                max_containment = 0.0
                
                # masks_3d shape: [N_points, N_masks] - each column is a mask
                for mask_idx in range(masks_3d.shape[1]):
                    mask_3d = masks_3d[:, mask_idx]  # Extract mask column [N_points]
                    # Check how much of the 3D mask is contained in this box
                    containment_ratio = self._compute_mask_containment_in_box(mask_3d, box, points)
                    max_containment = max(max_containment, containment_ratio)
                    
                    if containment_ratio > containment_threshold:
                        # This 3D mask is already well-covered by this box
                        # So this box is redundant
                        is_novel = False
                        break
                
                if is_novel:
                    # Convert to superpoint mask for merging
                    superpoint_overlap_threshold = box_filtering_config.get('superpoint_overlap', 0.3)
                    box_superpoint_mask = self._box_to_superpoint_mask(box, points, superpoints, superpoint_overlap_threshold)
                    filtered_frame_boxes.append({
                        **box,
                        'superpoint_mask': box_superpoint_mask,
                        'max_containment': max_containment
                    })
            
            filtered_boxes[frame_id] = filtered_frame_boxes
            total_filtered += len(filtered_frame_boxes)
        
        overall_ratio = total_filtered / total_original if total_original > 0 else 0
        self._log(f"[✅ INFO] Overall filtering: {total_original} → {total_filtered} boxes ({overall_ratio:.1%} kept)")
        
        return filtered_boxes
    
    def _compute_mask_containment_in_box(self, mask_3d, box_3d, scene_points):
        """
        Compute how much of a 3D mask is contained inside a 3D bounding box.
        
        Args:
            mask_3d: [N_points] binary mask column from 3D network
            box_3d: dict with 'box_center', 'box_size', 'box_rotation' or 'obb'
            scene_points: [N_points, 3] all scene point coordinates
            
        Returns:
            float: containment ratio between 0-1 (how much mask is inside box)
        """
        # Check if box has valid format first
        if not ('obb' in box_3d or ('box_center' in box_3d and 'box_size' in box_3d and 'box_rotation' in box_3d)):
            return 0.0  # Invalid box format, no containment
            
        # Get points that belong to the 3D mask
        if mask_3d.sum() == 0:
            return 0.0
            
        mask_points = scene_points[mask_3d.bool()]  # [N_mask_points, 3]
        
        # Check which mask points are inside the 3D box
        mask_points_in_box = self._get_points_in_oriented_box(mask_points, box_3d)
        
        # If no points were marked as inside (could be due to invalid box), return 0
        if mask_points_in_box.sum() == 0:
            return 0.0
        
        # Compute containment ratio
        containment_ratio = mask_points_in_box.sum().float() / len(mask_points)
        
        return containment_ratio.item()
    
    def _get_points_in_oriented_box(self, points, box_3d):
        """
        Check if points are inside an oriented 3D bounding box.
        
        Args:
            points: [N, 3] point coordinates
            box_3d: dict with either:
                - 'box_center', 'box_size', 'box_rotation' format, or
                - 'obb' format (Open3D OrientedBoundingBox with center, extent, R)
            
        Returns:
            torch.Tensor: [N] boolean mask of points inside box
        """
        if len(points) == 0:
            return torch.tensor([], dtype=torch.bool)
        
        # Handle different box formats
        if 'obb' in box_3d:
            # Extract from Open3D OrientedBoundingBox
            obb = box_3d['obb']
            center = torch.from_numpy(np.array(obb.center)).float()
            size = torch.from_numpy(np.array(obb.extent)).float()
            rotation = torch.from_numpy(np.array(obb.R)).float()
        elif 'box_center' in box_3d and 'box_size' in box_3d and 'box_rotation' in box_3d:
            # Handle standard format with both numpy and torch inputs
            if isinstance(box_3d['box_center'], np.ndarray):
                center = torch.from_numpy(box_3d['box_center']).float()
                size = torch.from_numpy(box_3d['box_size']).float()
                rotation = torch.from_numpy(box_3d['box_rotation']).float()
            else:
                center = box_3d['box_center'].float()
                size = box_3d['box_size'].float() 
                rotation = box_3d['box_rotation'].float()
        else:
            # Invalid box format - skip this box
            self._log(f"[⚠️ WARNING] Invalid 3D box format: missing required keys. Skipping box.", "WARNING")
            return torch.zeros(len(points), dtype=torch.bool)
        
        # Transform points to box coordinate system
        points_centered = points - center[None, :]  # [N, 3]
        points_local = points_centered @ rotation.T  # [N, 3]
        
        # Check if points are within box bounds
        half_size = size / 2
        inside = torch.all(
            (points_local >= -half_size[None, :]) & 
            (points_local <= half_size[None, :]), 
            dim=1
        )
        
        return inside
    
    def _box_to_superpoint_mask(self, box, points, superpoint_ids, overlap_threshold=0.3):
        """
        Convert a 3D bounding box to a superpoint-based binary mask.
        
        Args:
            box: 3D bounding box dict
            points: [N, 3] scene point coordinates
            superpoint_ids: [N] superpoint assignment for each point
            overlap_threshold: minimum overlap ratio to include a superpoint
            
        Returns:
            torch.Tensor: [N] binary mask for this box based on superpoints
        """
        # Find points inside the box
        points_in_box = self._get_points_in_oriented_box(points, box)
        
        if points_in_box.sum() == 0:
            return torch.zeros_like(superpoint_ids, dtype=torch.bool)
        
        # Find superpoints that overlap with the box
        superpoints_in_box = superpoint_ids[points_in_box]
        unique_superpoints = torch.unique(superpoints_in_box)
        
        # Filter superpoints by overlap ratio
        selected_superpoints = []
        for sp_id in unique_superpoints:
            sp_mask = (superpoint_ids == sp_id)
            sp_in_box_count = (sp_mask & points_in_box).sum()
            sp_total_count = sp_mask.sum()
            
            if sp_total_count > 0:
                overlap_ratio = sp_in_box_count.float() / sp_total_count.float()
                if overlap_ratio > overlap_threshold:
                    selected_superpoints.append(sp_id)
        
        # Create final mask from selected superpoints
        final_mask = torch.zeros_like(superpoint_ids, dtype=torch.bool)
        for sp_id in selected_superpoints:
            final_mask |= (superpoint_ids == sp_id)
        
        return final_mask
    
    def _load_ground_truth_instances(self, processed_scene):
        """
        Load ground truth instances for visualization.
        
        Args:
            processed_scene: Path to .npy file containing scene data
            
        Returns:
            dict: Ground truth instances data with masks and labels
        """
        if not GT_LOADING_AVAILABLE:
            self._log("[⚠️ WARNING] Ground truth loading not available - evaluation modules not found", "WARNING")
            return {}
        
        try:
            # Extract scene name from processed_scene path
            scene_name = osp.basename(osp.dirname(processed_scene))
            
            # Determine dataset type 
            if 'scannet200' in processed_scene:
                dataset_type = 'scannet200'
            elif 'replica' in processed_scene:
                dataset_type = 'replica'
            else:
                self._log(f"[⚠️ WARNING] Cannot determine dataset type from {processed_scene}", "WARNING")
                return {}
            
            # Construct ground truth file path
            # From: ./data/scannet200/scene0011_00/0011_00.npy
            # To:   ./data/scannet200/ground_truth/scene0011_00.txt
            data_root = osp.dirname(osp.dirname(processed_scene))  # ./data/scannet200
            gt_file = osp.join(data_root, 'ground_truth', f'{scene_name}.txt')
            
            if not osp.exists(gt_file):
                self._log(f"[⚠️ WARNING] Ground truth file not found: {gt_file}", "WARNING")
                return {}
            
            # Load ground truth instance IDs
            gt_ids = util_3d.load_ids(gt_file)
            
            # Create ID to label mapping
            ID_TO_LABEL = {}
            for i in range(len(VALID_CLASS_IDS)):
                ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            
            # Get instances using evaluation utility
            gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
            
            # Load scene points for mask creation
            scene_data = np.load(processed_scene)
            scene_points = scene_data[:, :3]  # xyz coordinates
            
            # Convert instances to masks
            instance_data = {}
            instance_counter = 0
            
            for class_label, instances in gt_instances.items():
                for instance in instances:
                    instance_id = instance['instance_id']
                    label_id = instance['label_id']
                    
                    # Create mask for this instance
                    instance_mask = (gt_ids == instance_id)
                    
                    if instance_mask.sum() > 0:  # Only include non-empty instances
                        instance_data[instance_counter] = {
                            'mask': instance_mask,
                            'points': scene_points[instance_mask],
                            'class_label': class_label,
                            'instance_id': instance_id,
                            'label_id': label_id,
                            'point_count': instance_mask.sum()
                        }
                        instance_counter += 1
            
            self._log(f"[✅ INFO] Loaded {len(instance_data)} ground truth instances from {scene_name}", "INFO")
            return instance_data
            
        except Exception as e:
            self._log(f"[❌ ERROR] Failed to load ground truth: {str(e)}", "ERROR")
            return {}
    
    def _generate_distinct_colors(self, num_colors):
        """
        Generate visually distinct colors for instance visualization.
        
        Args:
            num_colors: Number of distinct colors needed
            
        Returns:
            list: List of RGB color tuples (0-1 range)
        """
        if num_colors == 0:
            return []
            
        colors = []
        
        # Use HSV color space for better distribution
        for i in range(num_colors):
            # Distribute hues evenly around the color wheel
            hue = i / num_colors
            # Use high saturation and value for vibrant colors
            saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.9 + (i % 2) * 0.1       # Vary brightness slightly
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        # Shuffle to avoid similar colors being adjacent
        np.random.seed(42)  # For reproducible colors
        np.random.shuffle(colors)
        
        return colors
    
    def _compute_mask_iou(self, mask1, mask2):
        """
        Compute IoU (Intersection over Union) between two binary masks.
        
        Args:
            mask1: [N] boolean mask
            mask2: [N] boolean mask
            
        Returns:
            float: IoU score between 0-1
        """
        intersection = (mask1 & mask2).sum().float()
        union = (mask1 | mask2).sum().float()
        
        if union == 0:
            return 0.0
        
        return (intersection / union).item()
    
    def _merge_filtered_boxes_frame_by_frame(self, filtered_boxes):
        """
        Sequentially merge superpoint masks across frames with optional frame sampling.
        Each frame's boxes are merged with existing scene candidates.
        
        Args:
            filtered_boxes: dict of frame_id -> list of box dicts with superpoint_mask
            
        Returns:
            list: merged scene candidates with combined masks
        """
        scene_candidates = []
        box_filtering_config = self.openyolo3d_config.get('box_filtering', {})
        merge_threshold = box_filtering_config.get('merge_threshold', 0.4)
        frame_sampling = box_filtering_config.get('frame_sampling', 1)
        
        # Process all filtered frames (sampling was already done during filtering)
        all_frame_ids = sorted(filtered_boxes.keys())
        self._log(f"[🔍 DEBUG] Merging all {len(all_frame_ids)} filtered frames", "DEBUG")
        
        # Process frames in chronological order
        for frame_id in tqdm(all_frame_ids, desc="Merging frames"):
            frame_masks = filtered_boxes[frame_id]
            
            for mask_data in frame_masks:
                # Find best match among existing candidates
                best_match_idx = -1
                best_similarity = 0
                
                for candidate_idx, candidate in enumerate(scene_candidates):
                    similarity = self._compute_mask_similarity(
                        candidate['superpoint_mask'], 
                        mask_data['superpoint_mask'],
                        candidate, mask_data
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_idx = candidate_idx
                
                if best_similarity > merge_threshold:
                    # Merge with existing candidate
                    scene_candidates[best_match_idx] = self._merge_mask_candidates(
                        scene_candidates[best_match_idx], mask_data, frame_id
                    )
                else:
                    # Add as new candidate
                    scene_candidates.append({
                        **mask_data,
                        'support_frames': [frame_id],
                        'support_count': 1
                    })
        
        # Filter candidates by minimum support
        min_support_frames = box_filtering_config.get('min_support_frames', 1)
        stable_candidates = [c for c in scene_candidates if c['support_count'] >= min_support_frames]
        
        self._log(f"[🔍 DEBUG] Merged {len(scene_candidates)} candidates → {len(stable_candidates)} stable (min support: {min_support_frames})", "DEBUG")
        
        return stable_candidates
    
    def _compute_mask_similarity(self, mask1, mask2, meta1, meta2):
        """
        Compute similarity between two superpoint masks for merging decision.
        
        Args:
            mask1, mask2: torch.Tensor binary masks
            meta1, meta2: dict with metadata (label, score, etc.)
            
        Returns:
            float: similarity score for merging decision
        """
        # Spatial overlap (IoU)
        intersection = torch.logical_and(mask1, mask2).sum().float()
        union = torch.logical_or(mask1, mask2).sum().float()
        iou = intersection / union if union > 0 else 0
        
        # Semantic consistency
        semantic_match = (meta1['label'] == meta2['label'])
        confidence_factor = (meta1['score'] + meta2['score']) / 2
        
        # Size consistency (prevent merging very different sized objects)
        mask1_size = mask1.sum().float()
        mask2_size = mask2.sum().float()
        size_ratio = torch.min(mask1_size, mask2_size) / torch.max(mask1_size, mask2_size) if torch.max(mask1_size, mask2_size) > 0 else 0
        
        # Combine factors
        similarity = iou * (1 + 0.5 * semantic_match) * confidence_factor * size_ratio
        
        return similarity.item()
    
    def _merge_mask_candidates(self, existing, new, frame_id):
        """
        Merge two mask candidates into a single candidate.
        
        Args:
            existing: existing candidate dict
            new: new candidate dict  
            frame_id: current frame ID
            
        Returns:
            dict: merged candidate
        """
        return {
            **existing,
            'superpoint_mask': torch.logical_or(existing['superpoint_mask'], new['superpoint_mask']),
            'score': max(existing['score'], new['score']),
            'support_frames': existing['support_frames'] + [frame_id],
            'support_count': existing['support_count'] + 1,
            'max_containment': min(existing.get('max_containment', 1.0), new.get('max_containment', 1.0))
        }
    
    def _integrate_novel_candidates_with_3d_masks(self):
        """
        Convert merged novel candidates to binary masks and integrate with existing 3D network masks.
        Updates self.preds_3d to include both original 3D network masks and novel candidate masks.
        """
        if len(self.merged_novel_candidates) == 0:
            self._log("[ℹ️ INFO] No novel candidates to integrate")
            return
        
        # Get original 3D network masks and scores
        original_masks, original_scores = self.preds_3d  # [N_points, N_masks_orig]
        n_points = original_masks.shape[0]
        n_original_masks = original_masks.shape[1]
        n_novel_masks = len(self.merged_novel_candidates)
        
        self._log(f"[🔍 DEBUG] Integrating {n_novel_masks} novel masks with {n_original_masks} original 3D masks", "DEBUG")
        
        # Get NMS threshold and min points from config
        box_filtering_config = self.openyolo3d_config.get('box_filtering', {})
        nms_threshold = box_filtering_config.get('nms_threshold', 0.6)  # Dedicated NMS threshold
        min_points = box_filtering_config.get('min_points', 100)  # Minimum points per mask
        # Convert novel candidate superpoint masks and perform NMS against original masks
        novel_masks = []
        novel_scores = []
        nms_filtered_count = 0

        for candidate in self.merged_novel_candidates:
            # Extract superpoint mask and convert to proper format
            superpoint_mask = candidate['superpoint_mask']  # [N_points] boolean

            # Ensure it's the right shape and type
            if superpoint_mask.shape[0] != n_points:
                self._log(f"[⚠️ WARNING] Candidate mask shape mismatch: {superpoint_mask.shape[0]} vs {n_points} points", "WARNING")
                continue

            # Filter out masks with insufficient points
            if superpoint_mask.sum() < min_points:
                continue

            # Check IoU against all original 3D masks for NMS
            is_novel_after_nms = True
            max_iou = 0.0
            
            for mask_idx in range(n_original_masks):
                original_mask = original_masks[:, mask_idx].bool()  # [N_points] boolean
                iou = self._compute_mask_iou(superpoint_mask, original_mask)
                max_iou = max(max_iou, iou)
                
                if iou > nms_threshold:
                    # This novel candidate has high overlap with existing 3D mask
                    is_novel_after_nms = False
                    break
            
            if is_novel_after_nms:
                novel_masks.append(superpoint_mask.float())
                novel_scores.append(candidate['score'])
            else:
                nms_filtered_count += 1
                self._log(f"[🔍 DEBUG] NMS filtered novel candidate (IoU={max_iou:.3f} > {nms_threshold})", "DEBUG")
        
        if nms_filtered_count > 0:
            self._log(f"[📊 INFO] NMS filtered {nms_filtered_count}/{len(self.merged_novel_candidates)} novel candidates", "INFO")
            
        if len(novel_masks) == 0:
            self._log("[⚠️ WARNING] No novel masks remaining after NMS filtering", "WARNING")
            return
            
        # Stack novel masks into tensor [N_points, N_novel_masks]
        novel_masks_tensor = torch.stack(novel_masks, dim=1)  # [N_points, N_novel_masks]
        novel_scores_tensor = torch.tensor(novel_scores, dtype=original_scores.dtype)

        # Store separated masks for visualization
        self.original_3d_masks = (original_masks, original_scores)  # [N_points, N_original]
        self.novel_3d_masks = (novel_masks_tensor, novel_scores_tensor)  # [N_points, N_novel]

        # Combine original and novel masks
        combined_masks = torch.cat([original_masks, novel_masks_tensor], dim=1)  # [N_points, N_original + N_novel]
        combined_scores = torch.cat([original_scores, novel_scores_tensor], dim=0)  # [N_original + N_novel]

        # Update self.preds_3d with combined masks
        self.preds_3d = (combined_masks, combined_scores)

        self._log(f"[✅ INFO] Combined masks: {n_original_masks} original + {len(novel_masks)} novel = {combined_masks.shape[1]} total")

        # Store debug information
        self.n_original_masks = n_original_masks
        self.n_novel_masks = len(novel_masks)
    
    def _compute_novel_candidates(self, processed_scene, cache_path):
        """
        Compute novel candidates from scratch and save to cache.
        
        Args:
            processed_scene (str): Path to scene data
            cache_path (str): Path to save cached results
            
        Returns:
            list: Merged novel candidates
        """
        # Filter projected boxes against 3D masks
        self._log("[🚀 ACTION] Filtering projected boxes against 3D masks...")
        start = time.time()
        self.filtered_projected_boxes = self._filter_projected_boxes_with_3d_masks(processed_scene)
        self._log(f"[🕒 INFO] Filtering elapsed time {(time.time()-start)}")
        
        # Merge filtered boxes frame by frame
        self._log("[🚀 ACTION] Merging filtered boxes across frames...")  
        start = time.time()
        merged_candidates = self._merge_filtered_boxes_frame_by_frame(self.filtered_projected_boxes)
        self._log(f"[🕒 INFO] Merging elapsed time {(time.time()-start)}")
        self._log(f"[✅ INFO] Found {len(merged_candidates)} novel object candidates")
        
        # Save to cache
        if len(merged_candidates) > 0:
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            try:
                torch.save(merged_candidates, cache_path)
                self._log(f"[💾 INFO] Cached merged candidates to {cache_path}")
            except Exception as e:
                self._log(f"[⚠️ WARNING] Failed to save cached candidates: {e}", "WARNING")
        
        return merged_candidates
    
    def _setup_logging(self):
        """Set up logging configuration based on config settings."""
        logging_config = self.openyolo3d_config.get('logging', {})
        self.logging_enabled = logging_config.get('enabled', True)
        self.console_output = logging_config.get('console_output', True)
        
        if not self.logging_enabled:
            self.logger = None
            return
            
        # Create logger
        self.logger = logging.getLogger('OpenYolo3D')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set up formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        
        # Add console handler if enabled
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler will be added when we know the scene name
        self.log_dir = logging_config.get('log_dir', 'output/logs')
        self.file_handler = None
        
    def _setup_file_logging(self, scene_name):
        """Set up file logging once we know the scene name."""
        if not self.logging_enabled or self.file_handler is not None:
            return
            
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{scene_name}_{timestamp}.log"
        log_filepath = osp.join(self.log_dir, log_filename)
        
        # Add file handler
        self.file_handler = logging.FileHandler(log_filepath)
        self.file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        
        self.log_file_path = log_filepath
        if self.console_output:
            print(f"[📝 INFO] Logging to file: {log_filepath}")
    
    def _log(self, message, level="INFO"):
        """Log a message if logging is enabled."""
        if self.logging_enabled and self.logger:
            getattr(self.logger, level.lower())(message)
        elif not self.logging_enabled and self.console_output:
            # If logging disabled but console output enabled, still print to console
            print(message)
    
class WORLD_2_CAM():
    def __init__(self, path_2_scene, depth_scale, openyolo3d_config = None):
        self.poses = {}
        self.intrinsics = {}
        self.meshes = {}
        self.depth_maps_paths = {}
        self.depth_color_paths = {}
        self.vis_depth_threshold =  openyolo3d_config["openyolo3d"]['vis_depth_threshold']
        
        frequency = openyolo3d_config["openyolo3d"]['frequency']
        
        path_2_poses = osp.join(path_2_scene,"poses")
        num_frames = len(os.listdir(path_2_poses))
        self.poses = [osp.join(path_2_poses, f"{i}.txt") for i in list(range(num_frames))[::frequency]]
        
        path_2_intrinsics = osp.join(path_2_scene,"intrinsics.txt")
        self.intrinsics = [path_2_intrinsics for i in list(range(num_frames))[::frequency]] 
        
        self.mesh = sorted(glob.glob(path_2_scene+"/*.ply"))[0]
        path_2_depth = osp.join(path_2_scene,"depth")
        self.depth_maps_paths = [osp.join(path_2_depth, f"{i}.png") for i in list(range(num_frames))[::frequency]]
        
        path_2_color = osp.join(path_2_scene,"color")
        self.color_paths = [osp.join(path_2_color, f"{i}.jpg") for i in list(range(num_frames))[::frequency]]
        
            
        self.image_resolution = imageio.imread(list(self.color_paths)[0]).shape[:2]
        self.depth_resolution = imageio.imread(list(self.depth_maps_paths)[0]).shape
        self.height = self.depth_resolution[0]
        self.width = self.depth_resolution[1]
        self.depth_scale = depth_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @staticmethod
    def load_ply(path_2_mesh):
        pcd = o3d.io.read_point_cloud(path_2_mesh)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # print(points.shape)
        coords = np.concatenate([points, np.ones((points.shape[0], 1))], axis = -1)
        return coords, colors
    
    def load_depth_maps(self):
        depth_maps = []
        paths_to_depth_maps_scene_i = self.depth_maps_paths
        for depth_map_path_i in paths_to_depth_maps_scene_i:
            depth_path = os.path.join(depth_map_path_i)
            depth_maps.append(torch.from_numpy(imageio.imread(depth_path) / self.depth_scale).to(self.device))
        return torch.stack(depth_maps)
    
    def adjust_intrinsic(self, intrinsic, original_resolution, new_resolution):
        if original_resolution == new_resolution:
            return intrinsic
        
        resize_width = int(math.floor(new_resolution[1] * float(
                        original_resolution[0]) / float(original_resolution[1])))
        
        adapted_intrinsic = intrinsic.copy()
        adapted_intrinsic[0, 0] *= float(resize_width) / float(original_resolution[0])
        adapted_intrinsic[1, 1] *= float(new_resolution[1]) / float(original_resolution[1])
        adapted_intrinsic[0, 2] *= float(new_resolution[0] - 1) / float(original_resolution[0] - 1)
        adapted_intrinsic[1, 2] *= float(new_resolution[1] - 1) / float(original_resolution[1] - 1)
        return adapted_intrinsic
    
    def get_mesh_projections(self):
        N_Large = 2000000*250
        points, colors = self.load_ply(self.mesh)
        points, colors = torch.from_numpy(points).cuda(), torch.from_numpy(colors).cuda()
        
        intrinsic = self.adjust_intrinsic(np.loadtxt(self.intrinsics[0]), self.image_resolution, self.depth_resolution)
        intrinsics = torch.from_numpy(np.stack([intrinsic for frame_id in range(len(self.poses))])).cuda()
        extrinsics = torch.linalg.inv(torch.from_numpy(np.stack([np.loadtxt(pose) for pose in self.poses])).cuda())
        
        if extrinsics.shape[0]*points.shape[0] < N_Large:
            word2cam_mat = torch.einsum('bij, jk -> bik',torch.einsum('bij,bjk -> bik', intrinsics,extrinsics), points.T).permute(0,2,1)
        else:
            B_size = 800000
            Num_Points = points.shape[0]
            Num_batches = Num_Points//B_size+1
            word2cam_mat = []
            for b_i in range(Num_batches):
                dim_start = b_i*B_size
                dim_last = (b_i+1)*B_size if b_i != Num_batches-1 else points.shape[0]
                word2cam_mat_i = torch.einsum('bij, jk -> bik',torch.einsum('bij,bjk -> bik', intrinsics,extrinsics), points[dim_start:dim_last].T).permute(0,2,1)
                word2cam_mat.append(word2cam_mat_i.cpu())
            word2cam_mat = torch.cat(word2cam_mat, dim = 1)
        del intrinsics
        del extrinsics
        del points
        del colors
        torch.cuda.empty_cache()
        
        point_depth = word2cam_mat[:, :, 2].cuda()
        if word2cam_mat.shape[1]*word2cam_mat.shape[0] < N_Large:
            size = (word2cam_mat.shape[0], word2cam_mat.shape[1])
            mask = (word2cam_mat[:, :, 2] != 0).reshape(size[0]*size[1])
            
            projected_points = torch.stack([(word2cam_mat[:, :, 0].reshape(size[0]*size[1])[mask]/word2cam_mat[:, :, 2].reshape(size[0]*size[1])[mask]).reshape(size), 
                        (word2cam_mat[:, :, 1].reshape(size[0]*size[1])[mask]/word2cam_mat[:, :, 2].reshape(size[0]*size[1])[mask]).reshape(size)]).permute(1,2,0).long()
            inside_mask = ((projected_points[:,:,0] < self.width)*(projected_points[:,:,0] > 0)*(projected_points[:,:,1] < self.height)*(projected_points[:,:,1] >0) == 1 )
        
        else:
            B_size = 200000
            Num_Points = word2cam_mat.shape[1]
            Num_batches = Num_Points//B_size+1
            projected_points = []

            for b_i in range(Num_batches):
                dim_start = b_i*B_size
                dim_last = (b_i+1)*B_size if b_i != Num_batches-1 else word2cam_mat.shape[1]
                batch_z = word2cam_mat[:, dim_start:dim_last, 2].cuda()
                batch_y = word2cam_mat[:, dim_start:dim_last, 1].cuda()
                batch_x = word2cam_mat[:, dim_start:dim_last, 0].cuda()
                
                size = (word2cam_mat.shape[0], dim_last-dim_start)
                mask = (batch_z != 0).reshape(size[0]*size[1])
                projected_points_i = torch.stack([(torch.div(batch_x.reshape(size[0]*size[1])[mask],batch_z.reshape(size[0]*size[1])[mask])).reshape(size), 
                            (torch.div(batch_y.reshape(size[0]*size[1])[mask],batch_z.reshape(size[0]*size[1])[mask])).reshape(size)]).permute(1,2,0).long()
                projected_points.append(projected_points_i.cpu())
                

           
            # merge parts
            projected_points = torch.cat(projected_points, dim = 1)
            inside_mask = ((projected_points[:,:,0] < self.width)*(projected_points[:,:,0] > 0)*(projected_points[:,:,1] < self.height)*(projected_points[:,:,1] >0) == 1 )
            
        
        # Get visible points with depth, width, and height
        depth_maps = self.load_depth_maps()
        num_frames = depth_maps.shape[0]
        # pixel_to_3d_point = []
        for frame_id in range(num_frames):
            points_in_frame_mask = inside_mask[frame_id].clone()
            points_in_frame = (projected_points[frame_id][points_in_frame_mask])
            depth_in_frame = point_depth[frame_id][points_in_frame_mask]
            visibility_mask = (torch.abs(depth_maps[frame_id][points_in_frame[:,1].long(), points_in_frame[:,0].long()]
                                        - depth_in_frame) <= \
                                        self.vis_depth_threshold)
            
            inside_mask[frame_id][points_in_frame_mask] = visibility_mask.to(inside_mask.device)
        
        return projected_points.type(torch.int16).cpu(), inside_mask.cpu()
    

    def get_3d_bboxes_from_frame(self, frame_id: str, detections: dict):
        """
        Compute 3D bounding boxes from a single image frame and its 2D detections.

        Args:
            frame_id (str): ID of the image frame
            detections (dict): {'bbox': tensor(N,4), 'labels': tensor(N), 'scores': tensor(N)}

        Returns:
            tuple: (results, scene_pcd) where:
                - results: List of dicts with keys: 'points', 'obb', 'label', 'score'
                - scene_pcd: Full scene point cloud
        """
        dataset, scene_id = osp.dirname(self.mesh).split('/')[-2:]
        outdir = osp.join('output', dataset, '3d_projected', scene_id)
        os.makedirs(outdir, exist_ok=True)

        # --- Load intrinsics & extrinsics ---
        K = np.loadtxt(self.intrinsics[0]) 
        K = self.adjust_intrinsic(K, self.image_resolution, self.depth_resolution)
        pose_path = osp.join(osp.dirname(self.mesh), "poses", f"{frame_id}.txt")
        pose = np.loadtxt(pose_path)
        extrinsic = np.linalg.inv(pose)                # world <- cam

        depth_path = osp.join(osp.dirname(self.mesh), "depth", f"{frame_id}.png")
        depth = imageio.imread(depth_path) / self.depth_scale
        H, W = depth.shape
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        results = []
        save_results = []
        # Get bounding boxes and scale them to depth map resolution
        bboxes = detections['bbox'].cpu().numpy()
        bboxes[:, 0] = bboxes[:, 0] * self.depth_resolution[1] / self.image_resolution[1]  # scale x1
        bboxes[:, 2] = bboxes[:, 2] * self.depth_resolution[1] / self.image_resolution[1]  # scale x2
        bboxes[:, 1] = bboxes[:, 1] * self.depth_resolution[0] / self.image_resolution[0]  # scale y1
        bboxes[:, 3] = bboxes[:, 3] * self.depth_resolution[0] / self.image_resolution[0]  # scale y2

        labels = detections['labels'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = map(int, bboxes[i])
            label = labels[i]
            score = scores[i]
            
            # Ensure bbox coordinates are within image bounds
            x1, x2 = max(0, x1), min(W-1, x2)
            y1, y2 = max(0, y1), min(H-1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Create pixel grid inside the bbox
            us, vs = np.meshgrid(np.arange(x1, x2+1), np.arange(y1, y2+1))
            us, vs = us.flatten(), vs.flatten()
            
            try:
                ds = depth[vs, us]
                # Filter invalid depths and apply threshold
                valid = (ds > 0) & (ds < 10.0)  # 10m max depth threshold
                us, vs, ds = us[valid], vs[valid], ds[valid]
                
                if len(ds) < 50:  # Minimum points threshold
                    continue
                
                # Backproject to camera coordinates
                X = (us - cx) * ds / fx
                Y = (vs - cy) * ds / fy
                Z = ds
                pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # (N,4)
                
                # Transform to world coordinates
                pts_world = (pose @ pts_cam.T).T[:, :3]  # (N,3)
                
                # Create point cloud for this box
                box_pcd = o3d.geometry.PointCloud()
                box_pcd.points = o3d.utility.Vector3dVector(pts_world)
                
                # Compute oriented bounding box
                try:
                    obb = box_pcd.get_oriented_bounding_box()
                except:
                    obb = None 
                
                if obb is None: 
                    res = {
                        # "points": pts_world,
                        "label": label,
                        "score": score                    
                    }
                    
                    save_results.append(res)
                    results.append(res)                 
                else:
                    res = {
                        "box_center": obb.center,
                        "box_size": obb.extent,
                        "box_rotation": obb.R,
                        # "points": pts_world,
                        "label": label,
                        "score": score                    
                    }
                    
                    save_results.append(res)
                    results.append(res)

            except (IndexError, ValueError) as e:
                print(f"Warning: Error processing bbox {i}: {e}")
                continue
        save_file = osp.join(outdir, f"{frame_id}.pt")
        if not osp.exists(save_file):
            torch.save(save_results, save_file)
        return results

