#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve



def main(args):

    print("="*80, flush=True)
    print("进入 main 函数", flush=True)
    print("="*80, flush=True)

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)
        print(f"使用 CUDA 设备: {args.device_id}", flush=True)
    else:
        print("使用 CPU", flush=True)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    print(f"正在加载模型: {args.path}", flush=True)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    print("模型检查点加载成功", flush=True)
    
    task = tasks.setup_task(args)
    print("任务设置完成", flush=True)
    
    model = task.build_model(args)
    print("模型构建完成", flush=True)
    
    model.load_state_dict(state["model"], strict=False)
    print("模型状态字典加载完成", flush=True)

    # Move models to GPU
    if use_fp16:
        model.half()
        print("模型转换为 FP16", flush=True)
    if use_cuda:
        model.cuda()
        print("模型移至 GPU", flush=True)

    # Print args
    logger.info(args)

    print("="*80, flush=True)
    print(f"开始推理", flush=True)
    print("="*80, flush=True)

    model.eval()
    
    # 使用通用的推理函数,不依赖特定数据集
    print("执行前向推理...", flush=True)
    task.forward_inference(model)
    print("推理完成!", flush=True)
    
    print("="*80, flush=True)
    print("所有任务完成!", flush=True)
    print("="*80, flush=True)


def cli_main():
    # add args
    print("="*80, flush=True)
    print("开始执行 cli_main", flush=True)
    print("="*80, flush=True)
    

    parser = options.get_validation_parser()
    # 移除 test-task 参数,因为现在使用通用推理函数
    # parser.add_argument("--test-task", type=str, default="DUDE", help="test task", choices=["DUDE", "PCBA"])
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    print("调用 distributed_utils.call_main...", flush=True)
    
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
