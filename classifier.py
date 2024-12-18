import matplotlib.pyplot as plt
from PIL import Image

import cv2
import numpy as np
import os
import glob
######################################################################################
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from detect import *

import re
from unidecode import unidecode

# Function to normalize text


#######################################################################################
# init OCR
import easyocr

reader = easyocr.Reader(['vi','en'])


pre_define_CCCD = ['conghoaxahoichunghiavietnam',
 'doclap',
 'tudo',
 'hanhphuc',
 'socialistrepublicofvietnam',
 'independence',
 'freedom',
 'happiness',
 'cancuocongdan',
 'citizenidentitycard',
 'so/no']

total_check = 11

fix_image_size = [1080, 681]

height_position_percent = 0.73

width_percent_differ = 2.25
height_percent_differ = 0.95
############################################################################################


# Init YOLO v7

weights     = 'yolov7.pt'
source      = 'inference/images'  # file/folder, 0 for webcam
img_size    = 640
conf_thres  = 0.25  # object confidence threshold
iou_thres   = 0.45  # IOU threshold for NMS
device      = ''  # cuda device, i.e., '0', '0,1,2,3', or 'cpu'
view_img    = False  # display results
save_txt    = False  # save results to *.txt
save_conf   = False  # save confidences in --save-txt labels
nosave      = False  # do not save images/videos
classes     = None  # filter by class, e.g., [0] or [0, 2, 3]
agnostic_nms = False  # class-agnostic NMS
augment     = False  # augmented inference
update      = False  # update all models
project     = 'runs/detect'  # save results to project/name
name        = 'exp'  # save results to project/name
exist_ok    = True  # existing project/name ok, do not increment
no_trace    = False  # don't trace model


source, weights, view_img, save_txt, imgsz, trace = source, weights, view_img, save_txt, img_size, not no_trace
save_img = not nosave and not source.endswith('.txt')  # save inference images
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://', 'https://'))

# Directories
save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Initialize
set_logging()
device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, img_size)

if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

######################################################################################

def resize_image(input_path, output_path):
    # Đọc ảnh
    image = cv2.imread(input_path)
    
    width = 1080
    height = 651

    # Resize về kích thước mục tiêu
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Lưu ảnh đã resize
    cv2.imwrite(output_path, resized)
    #print(f"Ảnh đã resize và lưu tại: {output_path}")
    
    
def normalize_vietnamese(text):
    # Remove accents using unidecode
    text = unidecode(text)
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra spaces
    text = " ".join(text.split())
    text = text.replace(" ", "")
    
    return text
    
    
    
def detection():
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # if human == True:
            #     break
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    #print(s)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    #print(f"Image size (im0): {im0.shape}")
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        
                        if names[int(cls)] != "person":
                            continue
                        else:
                            face_bounding_box = [[int(xyxy[0]), int(xyxy[1])], [int(xyxy[2]), int(xyxy[3])]]
                            human = True
                            #print("Found Human")
                            return face_bounding_box, True
                            #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
    return None, False


def mainer(image_path):
    #image_path = "/home/phucnguyen/code/github_clone/YOLO/yolov7/inference/images/cccd_resized_standard.jpg"
    #image_path = "/home/phucnguyen/code/GIC/object_detection/data/QR/QR_2.jpg"
    

    resize_image(image_path, image_path)
    
    face_bounding_box, is_human = detection()
    
    if is_human != True:
        return "Else"
    

    face_bounding_box = face_bounding_box

    face_height = face_bounding_box[1][1] - face_bounding_box[0][1]
    face_width = face_bounding_box[1][0] - face_bounding_box[0][0]

    print("Check position: ", face_height, face_width )
    print("bounding box: ", face_bounding_box)
    
    if face_height > fix_image_size[1] * 2/3  or face_width > fix_image_size[0] * 1/3: # case face too big
        return "Human"
    elif  face_bounding_box[1][0] >= fix_image_size[0] /2  or face_bounding_box[0][1] < fix_image_size[1] * 1/3: #case face not in left side or on top side
        return "Human"
        
    #def calculate_wanted_area(fix_image_size, face_bounding_box, height_position_percent, width_percent_differ, height_percent_differ, face_height, face_width):
    text_bounding_box = [[0,0], [0,0]]

    text_bounding_box[1][0] = face_bounding_box[1][0] + face_width * width_percent_differ

    text_bounding_box[1][1] = face_bounding_box[1][1] - face_height * height_position_percent

    text_bounding_box[0][0] = face_bounding_box[1][0]
    text_bounding_box[0][1] = text_bounding_box[1][1] - face_height * height_percent_differ
    
    
    
    # Extract the coordinates
    top_left = text_bounding_box[0]
    bottom_right = text_bounding_box[1]

    # Calculate the other two corners
    top_right = [bottom_right[0], top_left[1]]  # [c, b]
    bottom_left = [top_left[0], bottom_right[1]]  # [a, d]

    # Combine all four corners
    all_corners = [top_left, top_right, bottom_right, bottom_left]

    
    
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        exit()

    # Function to crop regions based on bounding box
    output_dir = "crop/output_regions"
    os.makedirs(output_dir, exist_ok=True)


        # Convert points to numpy array
    points_array = np.array(all_corners, dtype=np.int32)

    # Get bounding rectangle (x_min, y_min, width, height)
    x, y, w, h = cv2.boundingRect(points_array)

    # Crop the region
    cropped = image[y:y+h, x:x+w]

    # Save the cropped region
    output_path = f"{output_dir}/cut.jpg"
    cv2.imwrite(output_path, cropped)
    
    
    result = reader.readtext('crop/output_regions/cut.jpg', detail = 0)
    result
    
    normalized_output = [normalize_vietnamese(item) for item in result]
    
    count = 0
    for word in normalized_output:
        if word in pre_define_CCCD:
            count +=1
            
    if count/11 > 0.5:
        print(count)
        return "Day la CCCD"
    else:
        return "Human"



if __name__ == '__main__':
    
    test_images = glob.glob("inference/test_data/*")
    for image_path in test_images:
        result = mainer(image_path)
        
        print("Image path: ", image_path)
        print(result)
                        
