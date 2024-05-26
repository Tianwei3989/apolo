# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

import yaml
from easydict import EasyDict as edict
import sys

import torch
import torch.nn as nn

from model.task_utils import (
    LoadDatasetEval,
    LoadLosses,
    EvaluatingModel,
)

from model.model import BertConfig
from model.model import WESD

import model.utils as utils
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="./data/pretrained_model.bin",
        type=str,
        help="the path to the trained model",
    )
    parser.add_argument(
        "--tasks", default="20", type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--w_clf", default=False, help="if the model is trained with classification loss"
    )
    parser.add_argument(
        "--output_dir",
        default="./data/results_art",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="/home/chentw/Downloads/vilbert-multi-task/config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument(
        "--use_chunk",
        default=0,
        type=float,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--batch_size", default=100, type=int, help="what is the batch size?"
    )
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument("--split", default="test", type=str, help="which split to use.")
    parser.add_argument(
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument(
        "--clean_train_sets",
        default=True,
        type=bool,
        help="whether clean train sets for multitask data.",
    )
    parser.add_argument(
        "--visual_target",
        default=0,
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )
    parser.add_argument(
        "--task_specific_tokens",
        action="store_true",
        help="whether to use task specific tokens for the multi-task learning.",
    )

    args = parser.parse_args()
    with open("./tasks.yml", "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_names = []
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        name = task_cfg[task]["name"]
        task_names.append(name)

    timeStamp = args.from_pretrained.split("/")[-1] + "-" + task_names[0]
    savePath = os.path.join(args.output_dir, timeStamp)
    config = BertConfig.from_json_file(args.config_file)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val = LoadDatasetEval(
        args, task_cfg, args.tasks.split("-")
    )

    model_series = str(args.from_pretrained).split('/')[-2] + "_" + args.split
    tbLogger = utils.tbLogger(
        timeStamp,
        savePath,
        task_names,
        task_ids,
        task_num_iters,
        1,
        save_logger=False,
        txt_name="eval_"+model_series+".txt",
    )

    if args.dynamic_attention:
        config.dynamic_attention = True
    if "roberta" in args.bert_model:
        config.model = "roberta"

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if args.task_specific_tokens:
        config.task_specific_tokens = True


    model = WESD(config=config)
    # load the trained weight
    state_dict = torch.load(args.from_pretrained, map_location="cpu")
    new_dict = {}
    for attr in state_dict:
        if attr.startswith("clf") or attr.startswith("fc_global"):
            new_dict[attr] = state_dict[attr]
    model.load_state_dict(new_dict)

    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)

    elif n_gpu > 1:
        model = nn.DataParallel(model)

    print("***** Running evaluation *****")
    print("  Num Iters: ", task_num_iters)
    print("  Batch size: ", task_batch_size)

    model.eval()
    # when run evaluate, we run each task sequentially.
    for task_id in task_ids:
        results = []
        others = []
        for i, batch in enumerate(task_dataloader_val[task_id]):
            loss, score, batch_size, results, others = EvaluatingModel(
                args,
                task_cfg,
                device,
                task_id,
                batch,
                model,
                task_dataloader_val,
                task_losses,
                results,
                others,
            )

            tbLogger.step_val(0, float(loss), float(score*100), task_id, batch_size, "val")

            sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
            sys.stdout.flush()

        if args.split:
            json_path = os.path.join(savePath, args.split)
        else:
            json_path = os.path.join(savePath, task_cfg[task_id]["val_split"])

        json.dump(results, open(json_path + "_result.json", "w"))
        json.dump(others, open(json_path + "_others.json", "w"))
        print('result saved at', json_path + "_result.json")


if __name__ == "__main__":

    main()
