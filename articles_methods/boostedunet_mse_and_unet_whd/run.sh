CUDA_VISIBLE_DEVICES=0 python hausdorff_net_detection_trainer.py --debug -dn certain_whd.json -sp splits/certain.json -mn certain_whd.bin
CUDA_VISIBLE_DEVICES=0 python detection_scorer.py -sp splits/certain.json -mn certain_whd.bin --ntree 1
CUDA_VISIBLE_DEVICES=0 python detection_infer.py -mn certain_whd.bin -ip ../certain/images/test -op whd/certain/inference
CUDA_VISIBLE_DEVICES=0 python detection_infer_with_mask.py -mn certain_whd.bin -ip whd/certain/inference -op whd/certain/inference