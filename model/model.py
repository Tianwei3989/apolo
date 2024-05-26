import copy
import json
import logging
import sys
from io import open

import torch
from torch import nn

logger = logging.getLogger(__name__)

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        v_feature_size=2048,
        v_target_size=1601,
        v_hidden_size=768,
        v_num_hidden_layers=3,
        v_num_attention_heads=12,
        v_intermediate_size=3072,
        bi_hidden_size=1024,
        bi_num_attention_heads=16,
        v_attention_probs_dropout_prob=0.1,
        v_hidden_act="gelu",
        v_hidden_dropout_prob=0.1,
        v_initializer_range=0.2,
        v_biattention_id=[0, 1],
        t_biattention_id=[10, 11],
        visual_target=0,
        fast_mode=False,
        fixed_v_layer=0,
        fixed_t_layer=0,
        in_batch_pairs=False,
        fusion_method="mul",
        dynamic_attention=False,
        with_coattention=True,
        objective=0,
        num_negative=128,
        model="bert",
        task_specific_tokens=False,
        visualization=False,
    ):

        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        assert len(v_biattention_id) == len(t_biattention_id)
        assert max(v_biattention_id) < v_num_hidden_layers
        assert max(t_biattention_id) < num_hidden_layers

        if isinstance(vocab_size_or_config_json_file, str) or (
            sys.version_info[0] == 2
            and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.v_feature_size = v_feature_size
            self.v_hidden_size = v_hidden_size
            self.v_num_hidden_layers = v_num_hidden_layers
            self.v_num_attention_heads = v_num_attention_heads
            self.v_intermediate_size = v_intermediate_size
            self.v_attention_probs_dropout_prob = v_attention_probs_dropout_prob
            self.v_hidden_act = v_hidden_act
            self.v_hidden_dropout_prob = v_hidden_dropout_prob
            self.v_initializer_range = v_initializer_range
            self.v_biattention_id = v_biattention_id
            self.t_biattention_id = t_biattention_id
            self.v_target_size = v_target_size
            self.bi_hidden_size = bi_hidden_size
            self.bi_num_attention_heads = bi_num_attention_heads
            self.visual_target = visual_target
            self.fast_mode = fast_mode
            self.fixed_v_layer = fixed_v_layer
            self.fixed_t_layer = fixed_t_layer

            self.model = model
            self.in_batch_pairs = in_batch_pairs
            self.fusion_method = fusion_method
            self.dynamic_attention = dynamic_attention
            self.with_coattention = with_coattention
            self.objective = objective
            self.num_negative = num_negative
            self.task_specific_tokens = task_specific_tokens
            self.visualization = visualization
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class WESD(nn.Module):
    def __init__(self, config, num_labels=1, dropout_prob=0.1, default_gpu=True):
        super().__init__()
        self.fc_global = nn.Linear(512, 512)
        self.clf = nn.Linear(512, 8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, txt_feature, img_feature, heatmap, segment_ids):
        emo_idx = segment_ids
        v_ = img_feature

        # add global feature before fc_clf
        v_g = self.fc_global(v_.permute(0, 2, 1).float().mean(axis=-1))
        v_g_ = torch.unsqueeze(v_g, 1).expand((-1, 7 * 7, img_feature.shape[-1]))
        v_fin = torch.add(v_, v_g_)

        # get feature from fc_clf (only visual feature as the input
        fea = self.clf(v_fin)
        fea_ = self.sigmoid(fea)

        # get the exact emotion prediction
        emo_idx_ = emo_idx.unsqueeze(1).expand((-1, v_.shape[1], fea_.shape[-1]))
        fea_emo = torch.sum(fea_.mul(emo_idx_),dim=-1)

        # get att map from CLIP & VinVL
        att_fea = heatmap.view(-1, v_.shape[1])
        fin_fea = fea_.permute(0, 2, 1)

        return fea_emo, fin_fea, fea, att_fea

