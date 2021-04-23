from rsmq.consumer import RedisSMQConsumer

import os
from shutil import copyfile
import time
import json
import datetime
import argparse
import tempfile
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from pprint import pprint

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random




from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box

from test_db_utils import DBManager


# import matplotlib.pyplot as plt
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
# plt.imshow(img)
# plt.show()


def plot_preds(img, im0, pred, frame, img_save_path, txt_save_path=None):
    # Process detections
    for i, det in enumerate(pred):  # detections per image

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if txt_save_path:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_save_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        cv2.imwrite(img_save_path, im0)


def detect(model, images_path, imgsz, device):
    stride = int(model.stride.max())  # model stride
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    dataset = LoadImages(images_path, img_size=imgsz, stride=stride)

    # out = model(torch.randn(1,3,416,416))[0]

    results = {}
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45)[0] # image by image, batch size is 1. it may be increased in future
        results[str(i)] = [det.tolist() for det in pred]

    return results



def write_predictions_to_db(pdf_original_path, predictions):
    global DB
    DB.add_detections(pdf_original_path, predictions)


def load_model(model_path, attributes_path, device):
    model = torch.jit.load(model_path)
    attrs = torch.load(Path(model_path).parent/'attributes.pt')
    model.stride = attrs['stride']
    model.names = attrs['names']
    model.to(device)
    model.eval()
    return model


def turn_pdf_to_pngs(pdf_path, pngs_dir):
    os.makedirs(pngs_dir)

    # transfer to pngs
    images = convert_from_path( # PIL images
        pdf_path,
        poppler_path=POPPLER_PATH,
    ) 

    for i, image in enumerate(images):
        image.save(pngs_dir/f"{i}.png")


def interpretate_predictions(label_names, predictions):
    return{ page: [{
        'x1': pred[0],
        'y1': pred[1],
        'x2': pred[2],
        'y2': pred[3],
        'class': label_names[np.argmax(pred[4:])]
    } for pred in predictions[page]] for page in predictions.keys()}
    

# define Processor
def processor(id, message, rc, ts):
    global POPPLER_PATH
    global DEVICE
    global IMG_RESIZE_SIZE
    global MODEL

    pdf_original_path = message['pdf_path']
    pdf_name = Path(pdf_original_path).name
    print(datetime.datetime.now(), f'started {pdf_original_path}')

    # check if pdf exists
    if not os.path.exists(pdf_original_path):
        # If it doesn't exist - then what can we do with it ?!
        return True


    # copy to temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir)/pdf_name
        pngs_dir = Path(temp_dir)/'pngs'
        copyfile(pdf_original_path, temp_file_path)


        turn_pdf_to_pngs(temp_file_path, pngs_dir)
        # TODO: may be refactored for converting PIL to cv2 directly, no files down 

        # detect
        predictions = detect(MODEL, pngs_dir, IMG_RESIZE_SIZE, DEVICE) # {'1':[], '2':[], ...}

    predictions = interpretate_predictions(MODEL.names, predictions)

    # write into a database
    write_predictions_to_db(pdf_original_path, predictions)    

    print(datetime.datetime.now(), f'done {pdf_original_path}')
    return True # The task will be closed


def run_worker():
    # create consumer
    consumer = RedisSMQConsumer('my-queue', processor, host='192.168.99.101')

    # run consumer
    consumer.run()


# may be None if poppler is in PATH. Else it should be to the poppler bin dir
model_path = r"D:\data\logos_and_signatures\pytorch_yolov5\runs\train\yolov5s_results6\weights\best.torchscript.pt"
attributes_path = r"D:\data\logos_and_signatures\pytorch_yolov5\runs\train\yolov5s_results6\weights\attributes.pt"

POPPLER_PATH = None
DEVICE = 'cpu'
IMG_RESIZE_SIZE = 416
MODEL = load_model(model_path, attributes_path, DEVICE)
DB = DBManager('results.sqlite')

def act1():
    temp_dir = Path('D:\data\logos_and_signatures')
    pdf_path = temp_dir / 'test_pdf_1.pdf'
    turn_pdf_to_pngs(pdf_path, temp_dir)

def act2():
    temp_dir = Path('D:\data\logos_and_signatures')
    pngs_dir = temp_dir/'pngs'
    predictions = detect(MODEL, pngs_dir, IMG_RESIZE_SIZE, DEVICE)
    predictions = interpretate_predictions(MODEL.names, predictions)
    pprint(predictions)

if __name__ == '__main__':
    # act1()
    # act2()
    run_worker()