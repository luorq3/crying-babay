import os
import shutil
from pathlib import Path
import numpy as np
from utils import save_img
from PIL import Image, ImageDraw, ImageFont
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords


class FaceDetect:
    def __init__(self, model_path, data_path, run_path, device, imgsz=(640, 640)):
        self.device = torch.device(device)
        self.run_path = run_path
        self.imgsz = imgsz
        self.model = DetectMultiBackend(model_path, device=self.device, data=data_path)

    def detect(self, im0, name):
        # Size scale
        im = letterbox(im0, self.imgsz)[0]
        # Convert
        im = im.transpose((2, 0, 1))  # HWC to CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, max_det=5)[0]  # Just one image, take the first pred

        img0 = Image.fromarray(im0)
        if len(pred):
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0.shape).round()
            img_draw = ImageDraw.ImageDraw(img0)
            for p in pred:
                confidence = p[4].item()
                face_left, face_top, face_length, face_width, _, det_cls = p.int()
                img_draw.rectangle(
                    ((face_left, face_top),
                     (face_length, face_width)),
                    outline='red',
                    width=5
                )
                img_draw.text((face_left, face_top - 10), str(round(confidence, 2)))
        save_img(name + '.jpg', img0, self.run_path)


if __name__ == '__main__':
    detect_model_path = 'yolov5/models/best.pt'
    data_path = 'datasets/baby-face/baby_config.yaml'
    run_path = 'datasets/baby-face-2/result'
    device = 'cpu'
    os.makedirs(run_path, exist_ok=True)
    face_detect = FaceDetect(detect_model_path, data_path, run_path, device)

    img_dir = 'datasets/baby-face-2'
    img_names = os.listdir(img_dir)
    test_imgs = np.random.choice(img_names, 200)
    print(f"Chosen {len(test_imgs)} test images.")
    for img_name in test_imgs:
        if img_name.split('.')[-1] not in ['png', 'jpg']:
            continue
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = img.convert('RGB')
        face_detect.detect(np.array(img), Path(img_name).stem)
