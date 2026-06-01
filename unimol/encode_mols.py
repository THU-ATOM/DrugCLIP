#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import torch
from unicore import distributed_utils, options, tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


# from skchem.metrics import bedroc_score


def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    # Load model
    # logger.info("loading model(s) from {}".format(args.path))
    # state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    # model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(f"encode mols args: {args}")

    model.eval()

    # names, scores = task.retrieve_mols(model, args.mol_path, args.pocket_path, args.emb_dir, 10000)

    # task.encode_mols_multi_folds(model, "/drug/DrugCLIP_chemdata_v2024/DrugCLIP_mols_v2024.lmdb", "/drug/tmp_save/")
    task.encode_mols_multi_folds(model, args.mol_path, args.save_dir, args.weight_path, args.airdd_test)


def cli_main():
    # add args

    parser = options.get_validation_parser()
    parser.add_argument("--save-dir", type=str, default="/data", help="save dir")
    parser.add_argument("--weight-path", type=str, default="", help="checkpoint weight pathr")
    parser.add_argument("--airdd-test", type=bool, default=False, help="do airdd test or not")

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    args.mol_path = os.path.join(args.save_dir, 'mol.lmdb')

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
