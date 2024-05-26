<p align="center">
         <img src="https://github.com/Tianwei3989/apolo/blob/main/images/annotation_examples.png" width="98%">
</p>

# APOLO: Artwork Provoked Emotion Evaluation
This project is about the study of Emotional Stimuli Retrieval in Artworks.

## APOLO Dataset
Please download the data associated with APOLO by filling this [form](https://forms.gle/9q2YaQV3V9hbdBCk9).
After approval, you will receive the following contents:
<details>
  <summary>Data list</summary>
  
     (dataset root) / apolo
    ├── artemis_index                         # index to retrive train, val, and test set from ArtEmis
    │   ├── train_index.json             
    │   ├── val_index.json    
    │   └── test_index.json   
    ├── apolo.json                            # index of APOLO annotation
    ├── pretrained_model.bin                  # pretrained WESD model
    ├── test_result.json                      # emotional stimuli prediction by the pretrained WESD model
    ├── apolo_pixel_map.zip                   # pixel-level annotations of APOLO
    ├── arts_features_vinvl_bbox_col.zip      # the bounding boxes predicted by VinVL
    ├── arts_features_clip.lmdb.zip           # the extracted feature from CLIP and VinVL, for training
    └── arts_features_clip_vinvl_heatmap.zip  # the extracted feature from CLIP and VinVL, for APOLO evaluation

</details>


To get the most out of this repo, please also download [ArtEmis](https://www.artemisdataset.org/#dataset) annotations.

We are sorry that due to the copyright requirements, we can not share images from WikiArt. We prepare the artwork features extracted from CLIP RN50 for the training process (The ``arts_features_clip.lmdb.zip``).
For visualization, please download the artworks images from the internet.

### Installation

Please run following code for installing the environment.
```commandline
conda create -n apolo python=3.6
conda activate apolo
git clone --recursive https://github.com/Tianwei3989/apolo.git
cd apolo
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

For training WESD on multiple GPU, please install [apex](https://github.com/NVIDIA/apex).

### Dataset preparation

Please follow the steps to prepare data for training and evaluation:

1. Doenload all files and move them to ``./data``
2. Prepare artwork images from WikiArt to ``./data/artworks`` with the name of ``artist-name_artwork-name``, e.g, ``william-merritt-chase_topaz-grapes``.
3. Download and extract ArtEmis dataset to get the key annotation file ``artemis_dataset_release_v0.csv``.
4. Extract ``arts_features_vinvl_bbox_col.zip`` to ``./data/arts_features_vinvl_bbox_col`` to get the bounding boxes predicted by VinVL.
4. Extract ``arts_features_clip.lmdb.zip`` to ``./data/arts_features_clip.lmdb`` to get the CLIP feature of the training.
5. Extract ``arts_features_clip_vinvl_heatmap.zip`` to ``./data/arts_features_vinvl_heatmap_sum_unified`` to get the CLIP + VinVL feature of the training
6. Extract ``arts_test_pixel_map.zip`` to ``./data/apolo_pixel_map`` to get the annotation of the APOLO.
6. Run ```python ./script/collect_train_val_test_data.py``` to get ``train.json``, ``val.json``, and ``test.json`` for WESD training process.
7. Run ```python ./script/recontruct_apolo.py``` to get ``apolo_val.json`` and ``apolo_test.json`` for apolo evaluation process.

### Visualization

Please use ``./notebooks/Visualize_apolo_from_artemis.ipynb`` to visualize the APOLO annotation.

Please note that you need to have the following files in advance: ``artemis_dataset_release_v0.csv``, ``apolo.json``, the artwork images in ``./data/artworks``, and the pixel-level annotations in ``./data/arts_features_clip_vinvl_heatmap``.

## WESD: Weakly-supervised Emotional Stimuli Detection

### Training
Please run ```python train_tasks.py``` to train WESD.
After this process, the trained models will be saved in ``./save/[model_saving_name]/pytorch_model_19.bin``.
By default, this code will use all of the existing GPU in your server. If you want to limit the number of GPU, please use ``CUDA_VISIBLE_DEVICES=[GPU_id]``.

### Evaluation 
Please follow the steps to evaluate WESD on the APOLO dataset.
1. Find the trained model from ``./save/[model_saving_name]/pytorch_model_19.bin``.
2. Run ```python ./eval_tasks.py --from_pretrained ./save/[model_saving_name]/pytorch_model_19.bin``` to get ``test_result.json``, which is the prediction of WESD on the artworks involved in APOLO.
3. Modify the path on Line 92 to ``test_result.json`` and run ```python ./eval_map_iou.py```. This code will directly print out the Pr@ by the end of the calculation.

### Visualization

Please use ``./notebooks/Visualize_WESD_prediction.ipynb`` to visualize the WESD prediction. you can also view other maps (e.g., the bounding boxes predict by VinVL) on this notebook.