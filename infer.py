from utils import pre_handle, save_img, increment_path
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords


class FaceResult(object):
    def __init__(self,
                 face_status,
                 face_top,
                 face_left,
                 face_length,
                 face_width,
                 face_direction,
                 is_masked,
                 confidence,
                 cropped,
                 annotated):
        self.face_status = face_status
        self.face_top = face_top
        self.face_left = face_left
        self.face_length = face_length
        self.face_width = face_width
        self.face_direction = face_direction
        self.is_masked = is_masked
        self.confidence = confidence
        self.cropped = cropped
        self.annotated = annotated


class FaceDetect:
    def __init__(self, model_path, data_path, device, imgsz=(640, 640)):
        self.device = torch.device(device)
        self.imgsz = imgsz
        self.model = DetectMultiBackend(model_path, device=self.device, data=data_path)

    def detect(self, im0):
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

        if len(pred):
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0.shape).round()
            detected_idx = torch.argmax(pred[:, 4])
            # choose the one with maximum confidence
            d = pred[detected_idx]
            confidence = d[4]
            face_left, face_top, face_length, face_width, _, det_cls = d.int()
            cropped_pic = im0[face_top:face_width, face_left: face_length]
            face_result = FaceResult(
                True,
                face_top.item(),
                face_left.item(),
                face_length.item(),
                face_width.item(),
                None,
                bool(det_cls),
                confidence.item(),
                Image.fromarray(cropped_pic, 'RGB'),
                Image.fromarray(im0, 'RGB')
            )
        else:
            face_result = FaceResult(False, *([None] * 9))

        return face_result


def attach_annotation(draw,
                      text,
                      xy,
                      wh,
                      ttf_path='./annotation_font/Louis_George_Cafe.ttf',
                      font_size=40,
                      color=(255, 0, 0)):
    # annotation font
    font = ImageFont.truetype(ttf_path, font_size)
    text_len = font.getlength(text)
    if xy[0] + text_len > wh[0]:
        x = wh[0] - text_len
        xy = (x, xy[1])
    draw.text(xy, text, font=font, fill=color)


def annotate(face_result, run_path):
    font_size = 30
    # annotate cropped image
    img_draw = ImageDraw.Draw(face_result.cropped)
    attach_annotation(img_draw,
                      face_result.face_direction,
                      (10, 0),
                      (face_result.cropped.width, face_result.cropped.height),
                      font_size=font_size)
    save_img("cropped.jpg", face_result.cropped, run_path)

    # annotate original image
    annotated_pic = face_result.annotated
    annotate_draw = ImageDraw.ImageDraw(annotated_pic)
    annotate_draw.rectangle(
        ((face_result.face_left, face_result.face_top),
         (face_result.face_length, face_result.face_width)),
        outline='red' if face_result.is_masked else 'green',
        width=5
    )
    if face_result.face_top >= font_size:
        annotation_pos = (face_result.face_left, face_result.face_top - font_size)
    else:
        annotation_pos = (face_result.face_left, face_result.face_width)
    attach_annotation(
        annotate_draw,
        f"Mask={face_result.is_masked}, Ori={face_result.face_direction}",
        annotation_pos,
        (annotated_pic.width, annotated_pic.height),
        font_size=font_size
    )
    save_img("annotated.jpg", annotated_pic, run_path)


class TowardsClassify:
    def __init__(self, label_path, model_path, device):
        with open(label_path) as label_file:
            line = label_file.readline()
            self.labels = line.split(',')
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)

    def inference(self, face_result, run_path, show=False):
        # Had not detected a object
        if not face_result.face_status:
            return face_result
        img0 = face_result.cropped
        img = pre_handle(img0)
        img = img[None]

        im_in = torch.Tensor(img).to(self.device)
        im_out = self.model(im_in)
        out_idx = torch.argmax(im_out, dim=1)
        idx = out_idx.detach().cpu()[0]

        face_result.face_direction = self.labels[idx]

        # annotate result to the image and save
        annotate(face_result, run_path)

        if show:
            face_result.cropped.show()
            face_result.annotated.show()

        return face_result


class ModelComb:
    def __init__(self,
                 detect_model_path,
                 data_path,
                 classify_model_path,
                 label_path,
                 device,
                 imgsz=(640, 640)):
        """
        Combination of `FaceDetect` and `TowardsClassify`.

        :param detect_model_path: detect model path.
        :param data_path: description of detect model.
        :param classify_model_path: classify model path.
        :param label_path: labels of classification.
        :param device: `cpu` or `cuda`.
        :param imgsz: will adjust images size to `imgsz` to fit detect model.
        :returns: `FaceResult`.
        """
        self.device = device
        self.face_detect = FaceDetect(detect_model_path, data_path, device, imgsz)
        self.towards_classify = TowardsClassify(label_path, classify_model_path, device)

    def infer(self, img, run_path='./runs/exp'):
        run_path = Path(increment_path(run_path, exist_ok=False))
        run_path.mkdir(parents=True, exist_ok=True)
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        face_result = self.face_detect.detect(np.array(img))
        face_result = self.towards_classify.inference(face_result, run_path)
        return face_result
