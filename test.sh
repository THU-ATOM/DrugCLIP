results_path="${RESULTS_PATH:-./test}"  # 从环境变量读取，默认为 ./test
batch_size="${BATCH_SIZE:-8}"  # 从环境变量读取，默认为 8
weight_path="${WEIGHT_PATH:-checkpoint_best.pt}"  # 从环境变量读取，默认为 checkpoint_best.pt

# 移除 TASK 参数,现在使用通用推理函数
# TASK="${TASK:-DUDE}"  # 从环境变量读取，默认为 PCBA (DUDE or PCBA)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \