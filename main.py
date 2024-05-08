from ultralytics import YOLO
import threading
import queue
from queue import Queue
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
import time
import argparse


def video_prepare(video_queue:Queue,video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened:
        print("VIDEO ERROR")
        cap.release()
    c  = 0
    while True:

        ret,frame = cap.read()

        if not ret:break
        video_queue.put((frame,c))
        c+=1 

    cap.release()

    return c , fps


 
def thread_safe_predict(queue_lock, video_queue: Queue, map_lock, frames, stop_event):
    local_model = YOLO('yolov8s-pose.pt')
    while True:
        try:
            with queue_lock:
                frame,idx = video_queue.get(timeout=1)
            res = local_model.predict(frame,device='cpu')
            with map_lock:
                frames[idx] = res[0].plot()
            # print(f"send frame {idx} by thread {threading.get_ident()}")
            # time.sleep(0.0001)
        except queue.Empty:
                if stop_event.is_set():
                    print(f'работа завершена by thread {threading.get_ident()}')
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",type=str,default='in_video.mp4')
    parser.add_argument("--threads", type=int, default =1 )
    parser.add_argument("--name", type=str, default = 'predicted_video')
    args = parser.parse_args()


    video_queue = Queue()
    video_path = args.video
    cap,fps = video_prepare(video_queue,video_path)
    num_threads = args.threads
    queue_lock = threading.Lock()
    map_lock = threading.Lock()
    frames = {}
    stop_event = threading.Event()
    workers = []

    start = time.monotonic()
    
    for _ in range(num_threads):
        workers.append(threading.Thread(target=thread_safe_predict,args=(queue_lock, video_queue, map_lock, frames, stop_event)))

    for thr in workers:
        thr.start()

    for thr in workers:
        stop_event.set()
        thr.join()

    end = time.monotonic()
    print(f'time: {end-start}')

    height,width,_ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{args.name}.mp4",fourcc,fps,(width,height))
    for idx in range(cap):
        out.write(frames[idx])
    out.release()

    