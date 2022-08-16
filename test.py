import numpy as np
from infer import ModelComb

device = 'cpu'
detect_model_path = 'yolov5/models/best.pt'
data_path = 'datasets/baby-face/baby_config.yaml'
# face_detect = FaceDetect(detect_model_path, data_path, device)
img_path = 'datasets/baby-face/valid/test/WechatIMG1661.jpg'
# res = face_detect.detect(img_path)
labels_path = 'datasets/labels.txt'
classify_model_path = 'towards/models/towards-classify.pt'
# c = TowardsClassify(labels_path, classify_model_path, device)
# c.inference(res)

models = ModelComb(detect_model_path, data_path, classify_model_path, labels_path, device)
results = models.infer(img_path)
for k, v in results.__dict__.items():
    if isinstance(v, np.ndarray):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {v}")



