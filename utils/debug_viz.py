"""
Debug Visualization Utilities for 2D-to-3D Projection Analysis

This module provides interactive visualization tools for debugging the 2D-to-3D bounding box
projection process using pyviz3d for better interactivity and control.
"""

import os
import json
import numpy as np
import torch
import cv2
import imageio
import colorsys
from typing import Dict, List, Tuple, Any, Optional
import pyviz3d.visualizer as viz
from models.Mask3D.mask3d import load_mesh_or_pc


class DebugProjectionVisualizer:
    """
    Interactive visualization tool for debugging 2D-to-3D bounding box projection
    """
    
    def __init__(self, openyolo3d, scene_name: str, dataset: str, output_dir: str = "debug_output"):
        """
        Initialize the debug visualizer
        
        Args:
            openyolo3d: OpenYolo3D instance with completed prediction
            scene_name: Name of the scene being debugged
            dataset: Dataset name (scannet200 or replica)
            output_dir: Output directory for debug files
        """
        self.openyolo3d = openyolo3d
        self.scene_name = scene_name
        self.dataset = dataset
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract key data
        self.world2cam = openyolo3d.world2cam
        self.preds_2d = openyolo3d.preds_2d
        self.preds_3d = openyolo3d.preds_3d
        self.config = openyolo3d.openyolo3d_config
        self.final_prediction = getattr(openyolo3d, 'final_prediction', None)
        
        # Use precomputed projection data if available, otherwise compute
        if hasattr(openyolo3d, 'projection_debug_data'):
            self.projection_debug_data = openyolo3d.projection_debug_data
            print("[✅ DEBUG] Using precomputed projection analysis")
        else:
            print("[⚠️ DEBUG] Projection data not found, computing fresh...")
            self.projection_debug_data = self._compute_projection_data()
        
        # Load scene geometry
        self.scene_geometry = self._load_scene_geometry()
        
        # Analyze projection quality
        self.projection_analysis = self._analyze_projection_quality()
        
        print(f"[🔍 DEBUG] Loaded scene with {len(self.preds_2d)} frames and {len(self.preds_3d[0][0])} 3D masks")
    
    def _load_scene_geometry(self) -> Dict[str, np.ndarray]:
        """Load scene point cloud or mesh"""
        mesh_path = self.world2cam.mesh
        
        if self.openyolo3d.datatype == "point cloud":
            pcd = load_mesh_or_pc(mesh_path, "point cloud")
            return {
                "points": np.asarray(pcd.points),
                "colors": np.asarray(pcd.colors),
                "type": "point cloud"
            }
        else:
            mesh = load_mesh_or_pc(mesh_path, "mesh")
            return {
                "vertices": np.asarray(mesh.vertices),
                "vertex_colors": np.asarray(mesh.vertex_colors),
                "triangles": np.asarray(mesh.triangles),
                "type": "mesh"
            }
    
    def _compute_projection_data(self) -> Dict[str, Any]:
        """Fallback method to compute projection data if not precomputed"""
        projection_data = {}
        
        for frame_id, detections in self.preds_2d.items():
            frame_idx = int(frame_id)
            
            # Compute 3D projections for this frame
            boxes_3d_results, scene_points = self.world2cam.get_3d_bboxes_from_frame(
                frame_idx, detections
            )
            
            projection_data[frame_id] = {
                'boxes_3d_results': boxes_3d_results,
                'scene_points': scene_points,
                'detections_2d': detections,
                'frame_idx': frame_idx
            }
        
        return projection_data
    
    def _analyze_projection_quality(self) -> Dict[str, Any]:
        """Analyze 2D-to-3D projections using precomputed data"""
        analysis = {
            "frames": {},
            "detection_stats": {},
            "projection_quality": {}
        }
        
        for frame_id, projection_data in self.projection_debug_data.items():
            detections = projection_data['detections_2d']
            boxes_3d_results = projection_data['boxes_3d_results']
            frame_idx = projection_data['frame_idx']
            
            # Analyze each detection
            frame_analysis = {
                "detections_2d": len(detections["bbox"]),
                "detections_3d": len(boxes_3d_results),
                "boxes_3d": [],
                "image_path": self.world2cam.color_paths[frame_idx],
                "depth_path": self.world2cam.depth_maps_paths[frame_idx]
            }
            
            # Detailed analysis for each 3D box
            for i, (box_2d, box_3d_data) in enumerate(zip(
                detections["bbox"], boxes_3d_results[:len(detections["bbox"])]
            )):
                if i < len(boxes_3d_results):
                    box_analysis = {
                        "id": i,
                        "frame_id": frame_id,
                        "bbox_2d": box_2d.cpu().numpy().tolist(),
                        "label": detections["labels"][i].item(),
                        "score": detections["scores"][i].item(),
                        "points_3d_count": len(box_3d_data["points"]),
                        "obb_center": box_3d_data["obb"].center.tolist(),
                        "obb_extent": box_3d_data["obb"].extent.tolist(),
                        "depth_stats": self._analyze_depth_distribution(box_3d_data["points"])
                    }
                    frame_analysis["boxes_3d"].append(box_analysis)
            
            analysis["frames"][frame_id] = frame_analysis
        
        return analysis
    
    def _analyze_depth_distribution(self, points_3d: np.ndarray) -> Dict[str, float]:
        """Analyze depth distribution of 3D points"""
        if len(points_3d) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        # Compute depth from origin
        depths = np.linalg.norm(points_3d, axis=1)
        
        return {
            "mean": float(np.mean(depths)),
            "std": float(np.std(depths)),
            "min": float(np.min(depths)),
            "max": float(np.max(depths))
        }
    
    def create_interactive_visualization(self, frame_limit: Optional[int] = None):
        """
        Create comprehensive interactive visualization with pyviz3d
        
        Args:
            frame_limit: Limit number of frames to visualize for performance
        """
        # Initialize visualizer
        v = viz.Visualizer()
        
        # Add scene geometry
        self._add_scene_geometry(v)
        
        # Add 2D-3D projection visualizations
        self._add_projection_visualizations(v, frame_limit)
        
        # Add metadata and controls
        self._add_metadata_display(v)
        
        # Save interactive visualization
        output_path = os.path.join(self.output_dir, f"{self.scene_name}_debug")
        v.save(output_path)
        
        # Generate analysis report
        self._generate_analysis_report()
        
        # Create 2D visualization overlay
        self._create_2d_overlay_visualization(frame_limit)
        
        print(f"[✅ SAVED] Interactive visualization: {output_path}/index.html")
        print(f"[✅ SAVED] Analysis report: {os.path.join(self.output_dir, 'analysis_report.json')}")
    
    def _add_scene_geometry(self, visualizer: viz.Visualizer):
        """Add scene point cloud or mesh to visualizer"""
        if self.scene_geometry["type"] == "point cloud":
            points = self.scene_geometry["points"]
            colors = self.scene_geometry["colors"] * 255  # Convert to 0-255 range
            
            visualizer.add_points(
                name="Scene",
                positions=points,
                colors=colors,
                point_size=25.0,
                visible=True
            )
        else:
            # For mesh, add vertices as points for now (pyviz3d mesh support is limited)
            vertices = self.scene_geometry["vertices"]
            vertex_colors = self.scene_geometry["vertex_colors"] * 255
            
            visualizer.add_points(
                name="Scene_Vertices",
                positions=vertices,
                colors=vertex_colors,
                point_size=25.0,
                visible=True
            )
    
    def _add_projection_visualizations(self, visualizer: viz.Visualizer, frame_limit: Optional[int]):
        """Add 2D-to-3D projection visualizations using precomputed data"""
        frame_ids = list(self.projection_debug_data.keys())
        if frame_limit:
            frame_ids = frame_ids[:frame_limit]
        
        colors = self._generate_vibrant_colors(len(frame_ids) * 10)  # Enough colors for all boxes
        color_idx = 0
        
        for frame_id in frame_ids:
            projection_data = self.projection_debug_data[frame_id]
            detections = projection_data['detections_2d']
            boxes_3d_results = projection_data['boxes_3d_results']
            
            # Add each 3D bounding box
            for i, box_3d_data in enumerate(boxes_3d_results):
                if i >= len(detections["bbox"]):
                    break
                
                # Get box metadata
                label = detections["labels"][i].item()
                score = detections["scores"][i].item()
                
                # Create box name with metadata
                box_name = f"Frame_{frame_id}_Box_{i}_Class_{label}_Score_{score:.2f}"
                
                # Add 3D points from this detection
                points_3d = box_3d_data["points"]
                if len(points_3d) > 0:
                    box_color = np.array([colors[color_idx % len(colors)]] * len(points_3d)) * 255
                    
                    visualizer.add_points(
                        name=box_name,
                        positions=points_3d,
                        colors=box_color,
                        point_size=35.0,
                        visible=True
                    )
                
                # Add oriented bounding box wireframe
                self._add_obb_wireframe(visualizer, box_3d_data["obb"], f"{box_name}_OBB", colors[color_idx % len(colors)])
                
                color_idx += 1
    
    def _add_obb_wireframe(self, visualizer: viz.Visualizer, obb, name: str, color: Tuple[float, float, float]):
        """Add oriented bounding box wireframe to visualizer"""
        # Get OBB vertices
        vertices = np.asarray(obb.get_box_points())
        
        # Define edges of a box (12 edges)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Create line segments for each edge
        lines = []
        line_colors = []
        for edge in edges:
            lines.extend([vertices[edge[0]], vertices[edge[1]]])
            line_colors.extend([color, color])
        
        lines = np.array(lines)
        line_colors = np.array(line_colors) * 255
        
        # Add lines to visualizer
        visualizer.add_lines(
            name=name,
            lines=lines.reshape(-1, 2, 3),
            colors=line_colors.reshape(-1, 2, 3),
            visible=True
        )
    
    def _add_metadata_display(self, visualizer: viz.Visualizer):
        """Add metadata and analysis information to visualization"""
        # Create summary statistics
        total_2d_detections = sum(len(dets["bbox"]) for dets in self.preds_2d.values())
        total_frames = len(self.preds_2d)
        total_3d_masks = len(self.preds_3d[0][0])
        
        # Add text information (if supported by pyviz3d version)
        metadata = {
            "scene": self.scene_name,
            "dataset": self.dataset,
            "total_frames": total_frames,
            "total_2d_detections": total_2d_detections,
            "total_3d_masks": total_3d_masks,
            "avg_detections_per_frame": total_2d_detections / total_frames if total_frames > 0 else 0
        }
        
        # Save metadata to separate file
        metadata_path = os.path.join(self.output_dir, "visualization_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_vibrant_colors(self, num_colors: int) -> List[Tuple[float, float, float]]:
        """Generate vibrant colors for visualization"""
        colors = []
        hue_increment = 1.0 / num_colors
        saturation = 1.0
        value = 1.0
        
        for i in range(num_colors):
            hue = i * hue_increment
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors
    
    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        report = {
            "scene_info": {
                "name": self.scene_name,
                "dataset": self.dataset,
                "total_frames": len(self.preds_2d),
                "total_3d_masks": len(self.preds_3d[0][0])
            },
            "detection_summary": {},
            "projection_analysis": self.projection_analysis,
            "quality_metrics": self._compute_quality_metrics()
        }
        
        # Detection statistics
        all_scores = []
        all_labels = []
        for detections in self.preds_2d.values():
            all_scores.extend(detections["scores"].cpu().numpy())
            all_labels.extend(detections["labels"].cpu().numpy())
        
        report["detection_summary"] = {
            "total_detections": len(all_scores),
            "score_statistics": {
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores))
            },
            "label_distribution": {int(label): int(count) for label, count in 
                                 zip(*np.unique(all_labels, return_counts=True))}
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _compute_quality_metrics(self) -> Dict[str, Any]:
        """Compute projection quality metrics"""
        metrics = {
            "projection_success_rate": 0.0,
            "average_points_per_box": 0.0,
            "depth_consistency": 0.0
        }
        
        total_boxes = 0
        successful_projections = 0
        total_points = 0
        depth_variations = []
        
        for frame_analysis in self.projection_analysis["frames"].values():
            for box in frame_analysis["boxes_3d"]:
                total_boxes += 1
                points_count = box["points_3d_count"]
                
                if points_count > 50:  # Threshold for successful projection
                    successful_projections += 1
                
                total_points += points_count
                depth_variations.append(box["depth_stats"]["std"])
        
        if total_boxes > 0:
            metrics["projection_success_rate"] = successful_projections / total_boxes
            metrics["average_points_per_box"] = total_points / total_boxes
        
        if depth_variations:
            metrics["depth_consistency"] = 1.0 / (1.0 + np.mean(depth_variations))  # Inverse of variation
        
        return metrics
    
    def _create_2d_overlay_visualization(self, frame_limit: Optional[int]):
        """Create 2D visualization with detection overlays"""
        frame_ids = list(self.preds_2d.keys())
        if frame_limit:
            frame_ids = frame_ids[:frame_limit]
        
        overlay_dir = os.path.join(self.output_dir, "2d_overlays")
        os.makedirs(overlay_dir, exist_ok=True)
        
        for frame_id in frame_ids:
            frame_idx = int(frame_id)
            image_path = self.world2cam.color_paths[frame_idx]
            detections = self.preds_2d[frame_id]
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Draw detections
            for i, (bbox, label, score) in enumerate(zip(
                detections["bbox"], detections["labels"], detections["scores"]
            )):
                x1, y1, x2, y2 = bbox.int().cpu().numpy()
                
                # Color based on score
                if score > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif score > 0.5:
                    color = (0, 165, 255)  # Orange for medium
                else:
                    color = (0, 255, 255)  # Yellow for low
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with metadata
                label_text = f"#{i} L{label.item()}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Background for text
                cv2.rectangle(image, 
                            (x1, y1 - text_height - 5), 
                            (x1 + text_width, y1), 
                            color, -1)
                
                # Text
                cv2.putText(image, label_text, 
                           (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 0), 2)
            
            # Save overlay
            output_path = os.path.join(overlay_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(output_path, image)
        
        print(f"[✅ SAVED] 2D overlay visualizations: {overlay_dir}")


class ProjectionQualityAnalyzer:
    """Analyze quality of 2D-to-3D projections"""
    
    def __init__(self, debug_visualizer: DebugProjectionVisualizer):
        self.debug_viz = debug_visualizer
    
    def analyze_projection_accuracy(self) -> Dict[str, Any]:
        """Analyze accuracy of 2D-to-3D projections"""
        # Implementation for projection accuracy analysis
        pass
    
    def detect_projection_anomalies(self) -> List[Dict[str, Any]]:
        """Detect potential projection anomalies"""
        # Implementation for anomaly detection
        pass
    
    def generate_quality_report(self) -> str:
        """Generate detailed quality assessment report"""
        # Implementation for quality report generation
        pass