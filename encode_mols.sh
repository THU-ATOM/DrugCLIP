results_path="./test"  # replace to your results path
batch_size=8
workdir=$1 # path to the save dir
checkpoint_dir=$2
weight_path="${checkpoint_dir}/checkpoint_best.pt"
airdd_test=$3


python3 /workspace/unimol/encode_mols.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --seed 1 \
       --log-interval 100 --log-format simple \
       --save-dir $workdir \
       --weight-path $weight_path \
       --airdd-test $airdd_test