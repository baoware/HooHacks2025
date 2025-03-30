# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0    # webcam, etc.
    (Other usage examples omitted for brevity)
"""
import send_alert
import argparse
import csv
import os
import platform
import sys
import serial
import time  # needed for alert timing and logging frequency
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5n.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.50,  # confidence threshold (only detections with >= 0.50 are processed)
    iou_thres=0.45,  # NMS IoU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, e.g., '0' or 'cpu'
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # format for saving coordinates
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class indices
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # directory to save results
    name="exp",  # subdirectory name
    exist_ok=False,  # if True, reuse existing directory
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Runs YOLOv5 detection inference on various sources.
    
    New functionality:
      - Only allowed classes ("person", "bicycle", "dog", "stop sign", "traffic light") are processed.
      - For "person" detections, an approximate distance is calculated and an alert is printed
        if a person is within 20m and at least 15 seconds have passed since the last alert.
      - A trapezoidal ROI is defined. For person detections, only those whose center falls inside the ROI are processed.
      - Only detections with a confidence of >= 0.50 are processed.
      - Inference logging is printed only every 10 frames.
    """
    # Allowed classes for processing
    static_classes = {"crosswalk", "stairs", "stop sign", "traffic light"}
    mobile_classes = {"person", "bicycle", "dog"}

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # Directory for saving results
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    
    # Alert parameters for person detection
    last_mobile_alert_time_far = 0  # timestamp of the last alert
    last_mobile_alert_time_close = 0
    last_static_alert_time_far = 0 
    last_static_alert_time_close = 0
    alert_interval_far = 20  # seconds between alerts
    alert_interval_close = 5
    distance_threshold_far = 20  # meters threshold for alert
    distance_threshold_close = 5 
    person_known_height = 1.7  # average person height in meters
    bicycle_known_height = 1.0
    dog_known_height = 0.8
    crosswalk_known_height = 0.2      # approximate height for a crosswalk marking (if using a physical barrier or flap; adjust as needed)
    stairs_known_height = 1.0         # approximate vertical extent of a staircase (flight height)
    stop_sign_known_height = 0.75     # approximate height/diameter of a stop sign
    traffic_light_known_height = 0.45 # approximate height of a traffic light housing
    focal_length = 1340       # example focal length in pixels

    frame_count = 0  # counter to log only every 10 frames

    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255  # normalize
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions for each image
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # Define trapezoidal ROI based on current frame dimensions
            h, w = im0.shape[:2]
            roi_points = np.array([
                [int(0.7 * w), int(0.1 * h)],
                [int(0.3 * w), int(0.1 * h)],
                [int(0.1 * w), h],
                [int(0.9 * w), h]
            ], np.int32)

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Process each detection
                for *xyxy, conf, cls in reversed(det):
                    # Only process detections with confidence >= 0.50
                    if conf < 0.50:
                        continue

                    c = int(cls)
                    # Only process allowed classes
                    if names[c] in mobile_classes:
                        x1, y1, x2, y2 = map(int, xyxy)
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        # Use OpenCV's pointPolygonTest to check if the center is inside the ROI
                        if cv2.pointPolygonTest(roi_points, center, False) < 0:
                            continue  # skip this detection if outside ROI
                        
                        # Compute approximate distance
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"
                        bbox_height = y2 - y1
                        height = 1
                        tag = names[c]
                        if tag == "person":
                            height = person_known_height
                        elif tag == "bicycle":
                            height = bicycle_known_height
                        elif tag == "dog":
                            height = dog_known_height
                        
                        # current_time = time.time()
                        # if current_time - last_mobile_alert_time >= alert_interval:
                        #    distance = (height * focal_length) / bbox_height if bbox_height > 0 else float('inf')
                        #    label_text = f"{tag}{confidence_str} {distance:.1f}m"
                        #    if distance <= distance_threshold:
                        #        print("Alert: Mobile object approaching within 20m!")
                        #        last_mobile_alert_time = current_time
                        # else:
                        #    continue

                        distance = (height * focal_length) / bbox_height if bbox_height > 0 else float('inf')
                        label_text = f"{tag} {confidence_str} {distance:.1f}m"
                        current_time = time.time()
                        
                        if distance <= distance_threshold_far:
                            if distance <= distance_threshold_close and (current_time - last_mobile_alert_time_close >= alert_interval_close):
                                send_alert.alert_low_close()
                                print("Alert: Mobile object within 5m!")
                                last_mobile_alert_time_close = current_time
                            elif (current_time - last_mobile_alert_time_far >= alert_interval_far):
                                send_alert.alert_low_far()
                                print("Alert: Mobile object approaching within 20m!")
                                last_mobile_alert_time_far = current_time
                        
                        

                    # If detection is "person", check if its center lies within the trapezoidal ROI
                    elif names[c] in static_classes:
                        x1, y1, x2, y2 = map(int, xyxy)
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        # Use OpenCV's pointPolygonTest to check if the center is inside the ROI
                        if cv2.pointPolygonTest(roi_points, center, False) < 0:
                            continue  # skip this detection if outside ROI
                        
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"
                        height = 1
                        tag = names[c]
                        if tag == "crosswalk":
                            height = crosswalk_known_height
                        elif tag == "stairs":
                            height = stairs_known_height
                        elif tag == "stop sign":
                            height = stop_sign_known_height
                        elif tag == "traffic light":
                            height = traffic_light_known_height

                        # Compute approximate distance
                        bbox_height = y2 - y1

                        # current_time = time.time()
                        # if current_time - last_static_alert_time >= alert_interval:
                        #    distance = (height * focal_length) / bbox_height if bbox_height > 0 else float('inf')
                        #    label_text = f"{tag} {confidence_str} {distance:.1f}m"
                        #    if distance <= distance_threshold:
                        #        print("Alert: Mobile object approaching within 20m!")
                        #        last_mobile_alert_time = current_time
                        # else:
                        #    continue

                        distance = (height * focal_length) / bbox_height if bbox_height > 0 else float('inf')
                        label_text = f"{tag} {confidence_str} {distance:.1f}m"
                        current_time = time.time()

                        if distance <= distance_threshold_far:
                            if distance <= distance_threshold_close and (current_time - last_static_alert_time_close >= alert_interval_close):
                                send_alert.alert_medium_close()
                                print("Alert: Static object within 5m!")
                                last_static_alert_time_close = current_time
                            elif (current_time - last_static_alert_time_far >= alert_interval_far):
                                send_alert.alert_medium_far()
                                print("Alert: Approaching static object in 20m!")
                                last_static_alert_time_far = current_time

                    else:
                        continue
                        # label_text = names[c] if hide_conf else f"{names[c]} {confidence_str}"


                    annotator.box_label(xyxy, label_text, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Draw the ROI on the frame so you can see the area being processed
            cv2.polylines(im0, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Log every 10 frames only
            if frame_count % 5 == 0:
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, e.g., '0' or 'cpu'")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-format", type=int, default=0, help="0: YOLO format, 1: Pascal-VOC format")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: e.g., --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand if only one value provided
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
