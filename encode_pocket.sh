batch_size=2
pocket_dir=$1 # path to the pocket dir
checkpoint_dir=$2
weight_path="${checkpoint_dir}/checkpoint_best.pt"
airdd_test=$3


python3 /workspace/unimol/encode_pockets.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --num-workers 1 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --seed 1 \
       --log-interval 100 --log-format simple \
       --pocket-dir $pocket_dir \
       --weight-path $weight_path \
       --airdd-test $airdd_test