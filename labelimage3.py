import os
import sys
import yaml
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

class AutoAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LabelKraft - Smart Image Annotation")
        self.root.configure(bg="#1e1e2f")  # Set background color

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2f")
        style.configure("TLabel", background="#1e1e2f", foreground="#ffffff", font=("Segoe UI", 10))
        style.configure("TButton", background="#3498db", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#2980b9")])
        style.configure("TCombobox", fieldbackground="#ffffff", background="#2c3e50", foreground="#000000")

        self.image_index = 0
        self.image_list = []
        self.annotations = []
        self.annotation_dict = {}
        self.current_image_path = None
        import sys
        from pathlib import Path
        model_path = Path(getattr(sys, '_MEIPASS', Path('.'))) / "yolov8n.pt"
        self.model = YOLO(str(model_path))
        self.detected_boxes = []
        self.tk_img = None
        self.selected_bbox = None

        self.status_var = tk.StringVar()

        self.setup_ui()

def setup_ui(self):
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="#0f111a")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        label_frame = ttk.Frame(self.root)
        label_frame.pack(pady=5)

        ttk.Label(label_frame, text="Label: ").pack(side=tk.LEFT)
        self.label_var = tk.StringVar()
        self.label_dropdown = ttk.Combobox(label_frame, textvariable=self.label_var)
        self.label_dropdown.pack(side=tk.LEFT)

        ttk.Button(label_frame, text="Save Label", command=self.save_label).pack(side=tk.LEFT, padx=5)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)

        ttk.Button(control_frame, text="Load Images", command=self.load_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export YAML", command=self.manual_export_yaml).pack(side=tk.LEFT, padx=5)

        status_frame = ttk.Frame(self.root)
        status_frame.pack(pady=5)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Segoe UI", 10))
        self.status_label.pack()

        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)

        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
