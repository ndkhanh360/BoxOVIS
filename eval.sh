python run_evaluation.py --dataset_name scannet200 --path_to_3d_masks "./output/scannet200/scannet200_masks" --path_to_2d_detections "./output/scannet200/2d_detections_with_logits" 
python run_evaluation.py --dataset_name scannet200 --path_to_2d_detections "./output/scannet200/2d_detections_with_logits" 

python run_evaluation.py --dataset_name replica --path_to_3d_masks "./output/replica/replica_masks" --path_to_2d_detections "./output/replica/2d_detections"
python run_evaluation.py --dataset_name replica --path_to_2d_detections "./output/replica/2d_detections"