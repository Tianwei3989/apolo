import os, cv2
import _pickle as cPickle
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import lmdb, pickle
import clip

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

LABEL_MAP = {"amusement": 0, "awe": 1, "contentment": 2, "excitement": 3,
             "anger": 4, "disgust": 5, "fear": 6, "sadness": 7}

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(item):

    if "emotion" in item.keys():
        emotion = item["emotion"]
    else:
        emotion = ''
    entry = {
        "question_id": item["question_id"],
        "image_id": item["image_id"],
        "emotion": emotion,
        "sentence": item["sentence"],
        "answer": item,
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'dev', 'test'
    """
    annotations_path = os.path.join(dataroot, "%s.json" % name)
    df = pd.read_json(annotations_path)
    # Build an index which maps image id with a list of hypothesis annotations.
    items = []
    count = 0

    for i in range(len(df)):
        dictionary = {}
        dictionary["id"] = df.iloc[i, :]["painting"]
        dictionary["image_id"] = df.iloc[i, :]["painting"]
        dictionary["emotion"] = df.iloc[i, :]["emotion"]
        dictionary["question_id"] = df.iloc[i, :]["painting"]
        dictionary["sentence"] = str(df.iloc[i, :]["utterance"])
        dictionary["labels"] = [int(LABEL_MAP[df.iloc[i, :]["emotion"]])]
        dictionary["scores"] = [1.0]
        items.append(dictionary)
        count += 1

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


class ArtEmisAttCLIPVinVLDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        clean_datasets,
        padding_index=0,
        max_seq_length=16,
        max_region_num=37,
    ):
        super().__init__()
        self.task = task
        self.split = split
        self.num_labels = 8
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.dataroot = dataroot
        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_oct_" + split + "_" + str(max_seq_length) + ".pkl",
            )

        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            text = entry["sentence"][:self._max_seq_length]
            tokens = clip.tokenize(text).tolist()[0]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self.entries:
            a = np.asarray(entry["q_token"])
            question = torch.from_numpy(a)
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            answer = entry["answer"]
            labels = np.array(answer["labels"])
            scores = np.array(answer["scores"], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry["answer"]["labels"] = labels
                entry["answer"]["scores"] = scores
            else:
                entry["answer"]["labels"] = None
                entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features = image_features_reader(image_id, self.dataroot)

        emotion = entry["emotion"]
        heatmap = heatmap_reader_u(image_id, emotion, self.dataroot)
        output_id = question_id + "_" + emotion

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))

        answer = entry["answer"]
        scores = answer["scores"]

        target = torch.zeros(self.num_labels)
        labels = answer["labels"]
        if labels is not None:
            target.scatter_(0, labels, scores)

        segment_ids = np.zeros(self.num_labels)
        segment_ids[int(answer["labels"])] = 1

        return (
            features,
            heatmap,
            features,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            output_id,
        )

    def __len__(self):
        return len(self.entries)

def image_features_reader(image_id, data_root):
    features_path = os.path.join(data_root, 'arts_features_clip.lmdb')
    env = lmdb.open(
        features_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:
        _image_ids = pickle.loads(txn.get("keys".encode()))

    image_id = str(image_id).encode()
    with env.begin(write=False) as txn:
        item = pickle.loads(txn.get(image_id))
        features = item["features"]

    return features

def heatmap_reader_u(art_name, emotion, data_root):
    features_path = './data/arts_features_vinvl_heatmap_sum_unified'
    heatmap = np.load(os.path.join(features_path, art_name + '_' + emotion + '.npy'), allow_pickle=True)
    heatmap_ = cv2.resize(heatmap, (7, 7), interpolation=cv2.INTER_AREA)
    heatmap_2 = (heatmap_ - np.min(heatmap_)) / (np.max(heatmap_) - np.min(heatmap_))
    return heatmap_2
