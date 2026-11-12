import torch
from tqdm import tqdm
import argparse
from evaluate import * # SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200, SCENE_NAMES_SCANNET200_DEBUG, evaluate_scannet200, evaluate_replica
from utils import OpenYolo3D
import yaml
import os.path as osp
import os
import time
import logging
from datetime import datetime
import shutil 

def setup_evaluation_logging(dataset_type, config, debug=False):
    """Set up logging for evaluation runs with structured directory including timestamp."""
    logging_config = config.get('logging', {})
    logging_enabled = logging_config.get('enabled', True)
    console_output = logging_config.get('console_output', True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = "debug" if debug else "all"

    if not logging_enabled:
        # Even if logging disabled, return the structured directory for output files
        output_dir = f"output/results/{dataset_type}/{mode}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return None, output_dir

    # Create logger
    logger = logging.getLogger('OpenYolo3D_Evaluation')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Set up formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')

    # Add console handler if enabled
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Create structured log directory with timestamp: output/results/<dataset>/<mode>/<timestamp>/
    log_dir = f"output/results/{dataset_type}/{mode}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    log_filename = f"evaluation_{dataset_type}_{mode}_{timestamp}.log"
    log_filepath = osp.join(log_dir, log_filename)

    # Add file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        print(f"[📝 INFO] Evaluation logging to file: {log_filepath}")
        print(f"[📁 INFO] Output directory: {log_dir}")

    return logger, log_dir

class InstSegEvaluator():
    def __init__(self, dataset_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type

    def evaluate_full(self, preds, scene_gt_dir, dataset, output_file='temp_output.txt', pretrained_on_scannet200=True):
        if dataset == "replica":
            inst_AP = evaluate_replica(preds, scene_gt_dir, output_file=output_file, dataset=dataset)
        elif dataset == "scannet200":
            inst_AP = evaluate_scannet200(preds, scene_gt_dir, output_file=output_file, dataset=dataset, pretrained_on_scannet200 = pretrained_on_scannet200)
        else:
            print("DATASET NOT SUPPORTED!")
            exit()
        return inst_AP

def test_pipeline_full(dataset_type, path_to_3d_masks, is_gt, debug=False, path_to_2d_detections=None, save_2d_detections=False):
    # Handle scannet200_debug as a subset of scannet200
    original_dataset_type = dataset_type
    is_debug_dataset = (dataset_type == "scannet200_debug") or ("class" in dataset_type)

    if is_debug_dataset:
        base_dataset_type = "scannet200"
    else:
        base_dataset_type = dataset_type
    
    debug_mode = debug

    # Use base dataset for all paths and configs
    config_path = osp.join(f'./pretrained/config_{base_dataset_type}.yaml')
    config = load_yaml(config_path)
    path_2_dataset = osp.join('./data', base_dataset_type)
    gt_dir = osp.join('./data', base_dataset_type, 'ground_truth')
    depth_scale = config["openyolo3d"]["depth_scale"]

    # Setup logging with original dataset name for proper directory structure
    logger, output_dir = setup_evaluation_logging(original_dataset_type, config, debug_mode)

    # Copy config file to log directory for reproducibility
    config_filename = f"config_{base_dataset_type}.yaml"
    config_dest = osp.join(output_dir, config_filename)
    shutil.copy2(config_path, config_dest)
    if logger:
        logger.info(f"Config file copied to: {config_dest}")
    print(f"[📋 INFO] Config file copied to: {config_dest}")

    # Create timestamped output file in structured directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = "debug" if debug_mode else "all"
    output_file = osp.join(output_dir, f"evaluation_results_{original_dataset_type}_{mode}_{timestamp}.txt")
    
    # Set scene names based on dataset type
    if base_dataset_type == "replica":
        scene_names = SCENE_NAMES_REPLICA
        datatype = "point cloud"
    elif base_dataset_type == "scannet200":
        if dataset_type == "scannet200_debug":
            scene_names = SCENE_NAMES_SCANNET200_DEBUG
        elif dataset_type == "scannet200":
            scene_names = SCENE_NAMES_SCANNET200
        else:
            scene_names = CLASS_TO_SCENE.get(dataset_type, [])
        datatype = "mesh"
    if debug:
        scene_names = scene_names[:1]

    # Apply additional debug mode (limit to 1 scene if --debug flag is used)
    if debug and not is_debug_dataset:
        scene_names = scene_names[:1]
        
    evaluator = InstSegEvaluator(base_dataset_type)
    openyolo3d = OpenYolo3D(f"./pretrained/config_{base_dataset_type}.yaml")
    predictions = {}
    processing_times = []  
    for scene_name in tqdm(scene_names):
        scene_id = scene_name.replace("scene", "")
        if base_dataset_type == "scannet200":
            processed_file = osp.join(path_2_dataset, scene_name, f"{scene_id}.npy")
        elif osp.exists(osp.join(path_2_dataset, scene_name, f"{scene_id}.npy")):
            processed_file = osp.join(path_2_dataset, scene_name, f"{scene_name}.npy")
        else:   
            processed_file = None
        t_start = time.time()
        prediction = openyolo3d.predict(path_2_scene_data = osp.join(path_2_dataset, scene_name), 
                                        depth_scale = depth_scale,
                                        datatype = datatype, 
                                        processed_scene = processed_file,
                                        path_to_3d_masks = path_to_3d_masks,
                                        is_gt = is_gt,
                                        path_to_2d_detections = path_to_2d_detections,
                                        save_2d_detections=save_2d_detections)
        t_end = time.time()
        processing_times.append(t_end-t_start)        
        predictions.update(prediction)
    print("[INFO] Avg processing time per scene:", sum(processing_times) / len(processing_times))    
    preds = {}
    print("Evaluation ...")
    for scene_name in tqdm(scene_names):
        preds[scene_name] = {
            'pred_masks': predictions[scene_name][0].cpu().numpy(),
            'pred_scores': torch.ones_like(predictions[scene_name][2]).cpu().numpy(), 
            # 'pred_scores': predictions[scene_name][2].cpu().numpy(), # let's test with output conf scores 
            'pred_classes': predictions[scene_name][1].cpu().numpy()}

    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=base_dataset_type, output_file=output_file)

    # Log completion info
    if logger:
        logger.info(f"Evaluation completed. Results saved to: {output_file}")
        logger.info(f"All output files saved in directory: {output_dir}")

    return inst_AP

def load_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='scannet200', type=str, help='Name of the dataset [replica, scannet200]')
    parser.add_argument('--path_to_3d_masks', default=None, type=str, help='Path to pre computed 3d masks')
    parser.add_argument('--is_gt', default=False, action=argparse.BooleanOptionalAction, help='If pre computed 3d masks are ground truth masks')
    parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction, help='Run debug/test mode')    
    parser.add_argument('--path_to_2d_detections', default=None, type=str, help='Path to precomputed 2D detections')
    parser.add_argument('--save_2d_detections', default=False, action=argparse.BooleanOptionalAction, help='Enable saving 2D detections')
    opt = parser.parse_args() 
    test_pipeline_full(opt.dataset_name, opt.path_to_3d_masks, opt.is_gt, debug=opt.debug, path_to_2d_detections=opt.path_to_2d_detections, save_2d_detections=opt.save_2d_detections)
