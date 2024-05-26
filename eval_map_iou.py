import cv2, os, imageio
import numpy as np
import pandas as pd
import json
from scipy.special import softmax
from scipy.special import expit
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", message="Lossy conversion from float64 to uint8")

imgs_path = './data/artworks/'

emotions = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
emotion_dict = {
    "amusement": 0, "awe": 1, "contentment": 2, "excitement": 3,
    "anger": 4, "disgust": 5, "fear": 6, "sadness": 7,
}

results_root = './data/'
result_code = '20'
save_file = False

# save_image = True
save_image = False

eval_bbox = False

output_path = results_root + 'art_pred'
if save_file:
    if not os.path.exists(os.path.join(output_path, result_code)):
        os.makedirs(os.path.join(output_path, result_code))

save_image_path = results_root + 'art_pred' + '/prediction_images/'
if save_image:
    if not os.path.exists(os.path.join(save_image_path, result_code)):
        os.makedirs(os.path.join(save_image_path, result_code))

def getMask_bbox(mask):
    '''
        input: df['mask'][i]
        output: object bounding box: (x1, y1, x2, y2)
    '''

    m = mask

    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]

    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        x2 += 1
        y2 += 1

    else:
        x1, x2, y1, y2 = 0, 0, 0, 0

    return np.array([x1, y1, x2, y2])

def graph2heatmap(g, fg, bg):
    T, mask = cv2.threshold((g+1).astype(np.uint8), 0.5, 255, cv2.THRESH_BINARY)
    num_labels, graph_3 = cv2.connectedComponents(mask.astype(np.uint8))

    graph_output = np.zeros(g.shape) + bg
    for i in range(1, np.max(graph_3) + 1):
        graph_t = np.where(graph_3 == i, 1, 0)
        g_bbox = getMask_bbox(graph_t)
        graph_output[g_bbox[1]:g_bbox[3], g_bbox[0]:g_bbox[2]] = fg

    return graph_output

def mIoU_(true: np.ndarray, pred: np.ndarray) -> float:
    if np.sum(pred + 1) == 0:
        return 0, 0
    c_matrix = confusion_matrix(true.ravel(), pred.ravel())
    intersection = np.diag(c_matrix)
    union = np.sum(c_matrix, axis=0) + np.sum(c_matrix, axis=1) - intersection
    iou = intersection / union
    miou = np.mean(iou)
    return miou, iou

#####  get annotation info
# answers_file_main = os.path.join('./data','apolo_val.json')
answers_file_main = os.path.join('./data','apolo_test.json')
df_anno = pd.read_json(answers_file_main)

##### label_map path
label_path = './data/apolo_pixel_map'

##### get prediction info
target_folder = '[path to test_result.json]'

prediction_file = '.save/'+target_folder+'/test_result.json'
f = open(prediction_file)
results = json.load(f)
df_result_ = pd.DataFrame(results)

df_res = pd.DataFrame()
missing_samples_id = []
emo_axis = 1
loc_axis = 0

pred_mtxes = []

for i in range(df_anno.shape[0]):
    utterance_id = (df_anno.iloc[i]['painting'] + '_' + df_anno.iloc[i]['emotion']).replace('.','#')
    sample_ = df_result_[df_result_.id == utterance_id]

    pred_mtx = sample_['prediction_matrix'].iloc[0]
    new_mtx_som = softmax(pred_mtx, axis=loc_axis)
    new_mtx_sig = expit(pred_mtx)

    pred_mtxes.append(new_mtx_sig)

df_res['pred_matrix'] = pred_mtxes


matrix_name = 'pred_matrix'

value_sum = 0.
correct_sum_1 = 0
correct_sum_2 = 0
counter = 0
threshold = 0.35

results = {
    'sample_idx': [],
    'painting': [],
    'emotion': [],
    'label_map_radio': [],
    'prediction_map_radio': [],
    'labelLpred': [],
    'score': [],
    'threshold': [],
}

for i in range(df_res.shape[0]):
    # decide sample and emotion
    sample_idx = i
    emotion_idx = emotion_dict[df_anno['emotion'].iloc[sample_idx]]

    # get mask from prediction
    a = np.asarray(df_res[matrix_name].iloc[sample_idx])
    b = a.T[emotion_idx].reshape([7, 7])
    c = (b - np.min(b)) / (np.max(b) - np.min(b))

    label_map = np.load(os.path.join(label_path,
                df_anno.iloc[sample_idx]['painting'] + '_' + df_anno.iloc[sample_idx]['emotion'] + '.npy'),
                             allow_pickle=True)
    mask_shape = label_map.shape
    prediction_map_inter = cv2.resize(c, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)
    prediction_map_linear = cv2.resize(c, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_LINEAR)

    # save as image
    if save_image:
        file_name = df_anno.iloc[sample_idx]['painting'] + '_' + df_anno.iloc[sample_idx]['emotion'] + '.jpg'
        img_fin_file = os.path.join(save_image_path, result_code, file_name)
        if not os.path.exists(img_fin_file):
            imageio.imwrite(img_fin_file, prediction_map_linear * 255)

        prediction_map_t = np.where(prediction_map_linear > threshold, 0, -1)

        if eval_bbox and np.sum(prediction_map_t) != 0:
            prediction_map_t_ = graph2heatmap(prediction_map_t, 0, -1)
            prediction_map_ = cv2.resize(prediction_map_t_, [mask_shape[1], mask_shape[0]], interpolation=cv2.INTER_NEAREST)
        else:
            prediction_map_ = prediction_map_t

    # get label map (only 1 for each sample)
    label_map_ = label_map - 1

    # skip NTL
    if np.sum(label_map_ + 1) == 0:
        print('Skip:', df_anno.iloc[sample_idx]['painting'] + '_' + df_anno.iloc[sample_idx]['emotion'] + ': NTL')
        continue

    mask = ((label_map_ >= 0) + (prediction_map_ >= 0)) & (label_map_ < 1) & (prediction_map_ < 1)

    score = mIoU_(label_map_[mask], prediction_map_[mask])[0] * 2

    if score >= 0.25:
        correct_sum_1 += 1

    if score >= 0.5:
        correct_sum_2 += 1

    label_map_radio = np.mean(label_map_+1)
    prediction_map_radio = np.mean(prediction_map_+1)

    print(str(sample_idx).zfill(4), f"{score:.10f}",
          "---", f"{label_map_radio:.10f}", f"{prediction_map_radio:.10f}", label_map_radio >= prediction_map_radio,
          str(threshold).zfill(2), df_anno.iloc[sample_idx]['painting'] + '_' + df_anno.iloc[sample_idx]['emotion'])

    save_code = 'bbox' if eval_bbox else 'seg'
    with open(output_path + '/' + result_code + '/' + \
              target_folder.split('.')[-1] + '_' + str(threshold).zfill(2) + '_'+save_code+'_val_log_230116.txt', 'a') as f:

        log_report = str(sample_idx).zfill(4) + ' ' + str(f"{score:.10f}") + ' ' + \
                    "---" + ' ' + str(f"{label_map_radio:.10f}") + ' ' + \
                    str(f"{prediction_map_radio:.10f}") + ' ' + \
                    str(label_map_radio >= prediction_map_radio) + ' ' + \
                    str(threshold).zfill(2) + ' ' + \
                    df_anno.iloc[sample_idx]['painting'] + '_' + df_anno.iloc[sample_idx]['emotion'] + '\n'

        f.write(log_report)

    if score >= 2:
        score = 1  # fix the bug in iou()

    value_sum += score
    counter += 1

print('-------------------------------------------')
print(target_folder.split('_')[-1])
print('mIOU:', value_sum / counter * 100)
print('Acc_0.25:', correct_sum_1 / counter * 100)
print('Acc_0.50:', correct_sum_2 / counter * 100)