import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from model.datasets import DatasetMapTrain, DatasetMapEval
from model.datasets._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "BCELoss": nn.BCELoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "MSELoss": nn.MSELoss(),
}


def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses):
    batch = batch[:-1]
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask = (batch)

    batch_size = features.size(0)

    vil_prediction, vil_prediction_gqa, vil_logit, _ = model(
        question,
        features,
        spatials,
        segment_ids,
    )

    feature = vil_prediction.view(-1, 7 * 7)  # from fea_emo, [bs, regions]
    heatmap = spatials.view(-1, 7 * 7)


    loss_map = task_losses[task_id](feature, heatmap)
    loss = loss_map.mean()

    batch_score = compute_score_with_logits(
        feature, heatmap
    ).sum() / float(batch_size)

    return float(loss), float(batch_score), batch_size


def ForwardModelsTrain(
    args,
    task_cfg,
    device,
    task_id,
    task_count,
    task_iter_train,
    task_dataloader_train,
    model,
    task_losses,
):

    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    # get the batch
    batch = task_iter_train[task_id].next()

    batch = batch[:-1]
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask = (
        batch
    )

    batch_size = features.size(0)

    vil_prediction, vil_prediction_gqa, vil_logit, _ = model(
        question,
        features,
        spatials,
        segment_ids,
    )

    feature = vil_prediction.view(-1, 7 * 7)  # from fea_emo, [bs, regions]
    heatmap = spatials.view(-1, 7 * 7)

    loss_map = task_losses[task_id](feature, heatmap)
    loss = loss_map.mean()

    batch_score = compute_score_with_logits(
        feature, heatmap
    ).sum() / float(batch_size)

    return loss, batch_score


def LoadLosses(args, task_cfg, task_ids):

    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def LoadDatasets(args, task_cfg, ids, split="trainval"):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_name = task_cfg[task]["name"]
        task_ids.append(task)
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
        num_workers = args.num_workers
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())

        # num_workers = int(num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        task_datasets_train[task] = None
        if "train" in split:
            task_datasets_train[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
                split=task_cfg[task]["train_split"],
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
            )

        task_datasets_val[task] = None
        if "val" in split:
            task_datasets_val[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
                split=task_cfg[task]["val_split"],
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
            if args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task])
            else:
                train_sampler = DistributedSampler(task_datasets_train[task])

            if args.local_rank == -1 and task == 'TASK12':
                train_sampler = RandomSampler(task_datasets_train[task],replacement=True)

            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task],
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if "val" in split:
            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                batch_size=batch_size,
                num_workers=2,
                pin_memory=True,
            )

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def LoadDatasetEval(args, task_cfg, ids):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_ids.append(task)
        task_name = task_cfg[task]["name"]
        batch_size = args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())

        num_workers = int(args.num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]["val_split"]

        task_datasets_val[task] = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=task_feature_reader1[
                task_cfg[task]["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg[task]["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            clean_datasets=args.clean_train_sets,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
        )

        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_val,
        task_dataloader_val,
    )


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def EvaluatingModel(
    args,
    task_cfg,
    device,
    task_id,
    batch,
    model,
    task_dataloader,
    task_losses,
    results,
    others,
):

    question_id = batch[-1]
    batch = batch[:-1]
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask = (batch)
    batch_size = features.size(0)

    with torch.no_grad():
        vil_prediction, vil_prediction_gqa, vil_logit, att_fea = model(
            question,
            features,
            spatials,
            segment_ids,
        )

        feature = vil_prediction.view(-1, 7 * 7)  # from fea_emo, [bs, regions]
        heatmap = spatials.view(-1, 7 * 7)

        loss_map = task_losses[task_id](feature, heatmap)
        loss = loss_map.mean()

        batch_score = compute_score_with_logits(
            feature, heatmap
        ).sum() / float(batch_size)

        feature_all = vil_logit

        for i in range(batch_size):
            results.append(
                {
                    'id': question_id[i],
                    'label': torch.argmax(target[i]).item(),
                    'answer': torch.argmax(target[i]).item(),  # same as label
                    'prediction_matrix': feature_all[i].cpu().numpy().tolist(),
                    'attention_feature': heatmap[i].cpu().numpy().tolist(),
                }
            )

    return float(loss), float(batch_score), batch_size, results, others
