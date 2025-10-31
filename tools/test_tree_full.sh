now=$(date +"%Y%m%d_%H%M%S")
root_path=/media/llog/AGMM-SASSdata/predict/0903_DINOV2b_full_84.36.pth
save_path=$root_path/full_84.36
resume_model=/home/lab532/llog/ours_method/exp/treecanopy/full0902_dinov2b/DINOV2b_84.36.pth
mkdir -p $save_path

python eval.py \
    --resume_model=$resume_model \
    --save-mask-path=$save_path | tee $save_path/$now.txt

python /home/lab532/llog/data_process/eval_accuracy/AGMM_eval.py \
    --folder_path=$save_path | tee -a $save_path/$now.txt
