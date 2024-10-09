import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import csv
import os

model = YOLO('yolov8n.pt')
deepfashion_model = YOLO('deepfashion2_yolov8s-seg.pt')

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fashion-Based-Person-Searcher")
        self.root.geometry("1680x1050")
        self.is_paused = False

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        self.load_button = tk.Button(self.control_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(self.control_frame, text="Stop Video", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT)
        
        self.pause_button = tk.Button(self.control_frame, text="Pause Video", command=self.pause_video)
        self.pause_button.pack(side=tk.LEFT)

        self.resume_button = tk.Button(self.control_frame, text="Resume Video", command=self.resume_video)
        self.resume_button.pack(side=tk.LEFT)
        
        self.save_button = tk.Button(self.control_frame, text="Save Data", command=self.save_data)
        self.save_button.pack(side=tk.LEFT)

        self.blank_label = tk.Label(self.control_frame, width=2)
        self.blank_label.pack(side=tk.LEFT)
        
        self.speed_label = tk.Label(self.control_frame, text="Playback Speed (1x = normal speed):")
        self.speed_label.pack(side=tk.LEFT)
        self.speed_scale = tk.Scale(self.control_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.speed_scale.set(1.0)
        self.speed_scale.pack(side=tk.LEFT)

        self.blank_label = tk.Label(self.control_frame, width=2)
        self.blank_label.pack(side=tk.LEFT)

        self.interval_label = tk.Label(self.control_frame, text="Capture Interval (seconds):")
        self.interval_label.pack(side=tk.LEFT)
        self.interval_scale = tk.Scale(self.control_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.interval_scale.set(1)
        self.interval_scale.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(root, width=800, height=600, bg='black')
        self.canvas.pack()

        self.scrollbar_frame = tk.Frame(root, height=700)
        self.scrollbar_frame.pack(fill="x", expand=False)

        self.image_list_canvas = tk.Canvas(self.scrollbar_frame, height=700)
        self.image_list_canvas.pack(side=tk.LEFT, fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self.scrollbar_frame, orient=tk.VERTICAL, command=self.image_list_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        self.image_list_frame = tk.Frame(self.image_list_canvas)
        self.image_list_canvas.create_window((0, 0), window=self.image_list_frame, anchor='nw')

        self.image_list_canvas.config(yscrollcommand=self.scrollbar.set)

        self.horizontal_scrollbar = tk.Scrollbar(self.scrollbar_frame, orient=tk.HORIZONTAL, command=self.image_list_canvas.xview)
        self.horizontal_scrollbar.pack(side=tk.BOTTOM, fill="x")

        self.image_list_canvas.config(xscrollcommand=self.horizontal_scrollbar.set)

        self.video_capture = None
        self.is_running = False
        self.detected_ids = {}
        self.next_id = 0

        self.frame_counter = 0
        self.fps = 0

        self.captured_data = []

        self.image_list_frame.bind("<Configure>", self.on_frame_configure)

    def on_frame_configure(self, event):
        self.image_list_canvas.configure(scrollregion=self.image_list_canvas.bbox("all"))

    def process_video(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.is_running = True
        self.update_frame()

    def update_frame(self):
        if self.video_capture.isOpened() and self.is_running and not self.is_paused:
            ret, frame = self.video_capture.read()
            if ret:
                interval = self.interval_scale.get()
                self.frame_counter += 1
    
                if self.frame_counter % int(self.fps * interval) == 0:
                    results = model(frame)

                    for result in results:
                        boxes = result.boxes.xyxy
                        confs = result.boxes.conf
                        labels = result.names

                        for box, conf, label in zip(boxes, confs, result.boxes.cls):
                            cls = int(label)

                            if cls == 0:
                                x1, y1, x2, y2 = map(int, box)
                                matched_id = self.match_bounding_box((x1, y1, x2, y2))

                                if matched_id is None:
                                    matched_id = self.next_id
                                    self.detected_ids[(x1, y1, x2, y2)] = matched_id
                                    self.next_id += 1
                                    self.save_detected_person(frame[y1:y2, x1:x2], matched_id, self.frame_counter / self.fps)
                                else:
                                    self.save_detected_person(frame[y1:y2, x1:x2], matched_id, self.frame_counter / self.fps, append=True)

                speed = self.speed_scale.get()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
                self.canvas.image = frame_tk

                delay = int(1000 / (self.fps * speed))
                self.root.after(delay, self.update_frame)
            else:
                self.video_capture.release()

    def match_bounding_box(self, bounding_box):
        for box, person_id in self.detected_ids.items():
            if self.calculate_iou(bounding_box, box) > 0.65:
                return person_id
        return None

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def save_detected_person(self, person_image, matched_id, capture_time, append=False):
        if person_image is not None and person_image.size > 0:
            person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
            person_image = Image.fromarray(person_image)

            width, height = person_image.size
            aspect_ratio = width / height
            new_height = 200
            new_width = int(new_height * aspect_ratio)
            person_image = person_image.resize((new_width, new_height))

            person_image_tk = ImageTk.PhotoImage(person_image)

            row = self.get_row_for_id(matched_id)
            label = tk.Label(row, image=person_image_tk)
            label.image = person_image_tk
            label.pack(side=tk.LEFT, padx=5, pady=5)

            clothing_info = self.get_clothing_info(person_image)

            info_label = tk.Label(row, text=f"ID: {matched_id} | Time: {capture_time:.2f}s | Clothing: {', '.join(clothing_info)}")
            info_label.pack(side=tk.LEFT, padx=5, pady=5)

            self.captured_data.append({
                'ID': matched_id,
                'Time': capture_time,
                'Clothing': ', '.join(clothing_info)
            })

            self.save_image_to_folder(person_image, matched_id)

            self.on_frame_configure(None)

    def save_image_to_folder(self, person_image, matched_id):
        folder_path = f"captured_images/{matched_id}"
        os.makedirs(folder_path, exist_ok=True)
        image_count = len(os.listdir(folder_path)) + 1
        image_path = os.path.join(folder_path, f"{image_count}.jpg")
        person_image.save(image_path)

    def get_row_for_id(self, matched_id):
        for child in self.image_list_frame.winfo_children():
            if child.winfo_name() == f"row_{matched_id}":
                return child

        new_row = tk.Frame(self.image_list_frame, name=f"row_{matched_id}")
        new_row.pack(fill=tk.X)
        return new_row

    def get_clothing_info(self, person_image):
        clothing_info = []
        results = deepfashion_model(person_image)

        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            labels = result.boxes.cls

            for box, conf, label in zip(boxes, confs, result.boxes.cls):
                clothing_info.append(result.names[int(label)])

        return clothing_info

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if video_path:
            self.stop_video()
            self.process_video(video_path)

    def stop_video(self):
        self.is_running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            self.canvas.delete("all")

    def pause_video(self):
        self.is_paused = True

    def resume_video(self):
        self.is_paused = False

    def save_data(self):
        with open('captured_data.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['ID', 'Time', 'Clothing'])
            writer.writeheader()
            writer.writerows(self.captured_data)

root = tk.Tk()
app = VideoApp(root)
root.mainloop()
