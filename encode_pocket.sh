batch_size=2
pocket_dir=$1 # path to the pocket dir

weight_path="/checkpoints/checkpoint_best.pt"


python3 /workspace/unimol/encode_pockets.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --num-workers 1 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --pocket-dir $pocket_dir