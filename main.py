#!/usr/bin/env python3

from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient
import numpy as np
import argparse

MODEL = "yolov8n.pt"
SOURCE = 0
OSC_HOST = "127.0.0.1"
OSC_PORT = 2345
SHOW = True
ADDRESS = "/poi"
SOURCE_WIDTH = 640
SOURCE_HEIGHT = 480


def poi(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    new_x = (x2 - x1) / 2.0 + x1
    new_y = (y2 - y1) / 3.0 + y1

    return [new_x / SOURCE_WIDTH, new_y / SOURCE_HEIGHT]


def args():
    global SHOW

    parser = argparse.ArgumentParser(prog="aide-vision")
    parser.add_argument("-s", "--show", action="store_true")

    results = parser.parse_args()
    SHOW = results.show


def main():
    args()
    model = YOLO(MODEL)
    osc = SimpleUDPClient(OSC_HOST, OSC_PORT)

    for results in model.track(source=SOURCE, show=SHOW, stream=True, classes=[0]):
        data = results.boxes.xyxy.cpu().numpy()
        if len(data) > 0:
            all_points = list(map(poi, data))
            points = x = [all_points[i : i + 2] for i in range(0, len(all_points), 2)]
            id = 0
            for point in points:
                address = f"{ADDRESS}/{id}"
                print(f"{id} = {point}")
                osc.send_message(address, point[0])
                id += 1


if __name__ == "__main__":
    main()
