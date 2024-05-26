from .artemis_clip_vinvl_dataset import ArtEmisAttCLIPVinVLDataset

__all__ = [
    "ArtEmisAttCLIPVinVLDataset",
    "",
]

DatasetMapTrain = {
    "WESD": ArtEmisAttCLIPVinVLDataset,
}


DatasetMapEval = {
    "WESD": ArtEmisAttCLIPVinVLDataset,
}
