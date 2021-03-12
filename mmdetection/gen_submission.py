import numpy as np
import pickle
import json
import pandas as pd

res = pickle.load(open('./0311/valid.pkl', 'rb'))
pred = []

for p in res:
    patient = []
    #tmp_score = np.array([b[-1] for i_class in range(14) for b in p[i_class]])
    #if np.any(tmp_score>0.5):
    for i_class in range(15):
        for box in p[i_class]:
                #if box[-1]>0.3:
            if len(box)>0:
                if i_class == 14:
                    print('Normal')
                    patient.append(f"14 1 0 0 1 1")
                    break
                else:
                    patient.append(f"{i_class} {box[-1]} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}")
    #else:
        #patient.append("14 1 0 0 1 1")
    pred.append(" ".join(patient))

test_json = json.load(open('./vinbigdata_coco_chest_xray_normal/valid_annotations.json', 'r'))
img_names = [p['file_name'].split('/')[-1].rstrip(".npz") for p in test_json['images']]

df = pd.DataFrame({'image_id': img_names, 'PredictionString': pred})
df.to_csv('./0311/valid.csv', index=False)
