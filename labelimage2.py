import os
import yaml
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

class AutoAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Image Annotation Tool")

        self.image_index = 0
        self.image_list = []
        self.annotations = []
        self.annotation_dict = {}
        self.current_image_path = None
        self.model = YOLO("yolov8n.pt")
        self.detected_boxes = []
        self.tk_img = None
        self.selected_bbox = None

        self.status_var = tk.StringVar()

        self.setup_ui()

    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        label_frame = tk.Frame(self.root)
        label_frame.pack(pady=5)

        tk.Label(label_frame, text="Label: ").pack(side=tk.LEFT)
        self.label_var = tk.StringVar()
        self.label_dropdown = ttk.Combobox(label_frame, textvariable=self.label_var)
        self.label_dropdown.pack(side=tk.LEFT)

        tk.Button(label_frame, text="Save Label", command=self.save_label).pack(side=tk.LEFT, padx=5)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5)

        tk.Button(control_frame, text="Load Images", command=self.load_images).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Export YAML", command=self.manual_export_yaml).pack(side=tk.LEFT, padx=5)

        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=5)
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.pack()

        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)

        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

    def load_images(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        self.image_list = [os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.image_list.sort()

        if not self.image_list:
            messagebox.showerror("Error", "No images found!")
            return

        self.image_index = 0
        self.annotations = []
        self.annotation_dict = {}
        self.update_status()
        self.show_image()

    def update_status(self):
        total = len(self.image_list)
        done = len(self.annotation_dict.keys())
        self.status_var.set(f"Total Images: {total} | Annotated: {done} | Remaining: {total - done}")
        if total > 0:
            self.progress['value'] = (done / total) * 100

    def show_image(self):
        if not self.image_list:
            return

        self.selected_bbox = None
        self.current_image_path = self.image_list[self.image_index]
        img = cv2.imread(self.current_image_path)
        results = self.model(img)[0]
        self.detected_boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_idx = int(box.cls[0].item())
            label = self.model.names[label_idx]
            self.detected_boxes.append({"bbox": (x1, y1, x2, y2), "label": label})

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for item in self.detected_boxes:
            x1, y1, x2, y2 = item["bbox"]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = cv2.putText(img, item["label"], (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        self.img_for_click = img.copy()
        img_pil = Image.fromarray(img)
        img_pil.thumbnail((640, 480))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(320, 240, image=self.tk_img)

        self.label_dropdown['values'] = list(set([item["label"] for item in self.detected_boxes]))
        if self.detected_boxes:
            self.label_var.set(self.detected_boxes[0]["label"])
            self.selected_bbox = self.detected_boxes[0]["bbox"]

        self.update_status()

    def on_canvas_click(self, event):
        canvas_width, canvas_height = 640, 480
        img = Image.open(self.current_image_path)
        img.thumbnail((canvas_width, canvas_height))
        scale_x = img.width / canvas_width
        scale_y = img.height / canvas_height
        x = int(event.x * scale_x)
        y = int(event.y * scale_y)

        for item in self.detected_boxes:
            x1, y1, x2, y2 = item["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.label_var.set(item["label"])
                self.selected_bbox = item["bbox"]
                return

    def save_label(self):
        label = self.label_var.get().strip()
        if not label or self.selected_bbox is None:
            messagebox.showwarning("Warning", "Please select a bounding box and label before saving.")
            return

        image_name = os.path.basename(self.current_image_path)
        entry = {
            "label": label,
            "bbox": list(map(int, self.selected_bbox))
        }

        if image_name not in self.annotation_dict:
            self.annotation_dict[image_name] = []

        if entry not in self.annotation_dict[image_name]:
            self.annotation_dict[image_name].append(entry)

        self.export_yaml()
        self.next_image()
        self.update_status()

    def next_image(self):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.show_image()
        else:
            messagebox.showinfo("Done", "All images processed!")
            self.update_status()

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.show_image()
            self.update_status()

    def export_yaml(self):
        data = {"annotations": []}
        for image, entries in self.annotation_dict.items():
            for entry in entries:
                data["annotations"].append({
                    "image": image,
                    "label": entry["label"],
                    "bbox": entry["bbox"]
                })

        try:
            with open("annotations.yaml", "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            print("Error writing YAML file:", e)

    def manual_export_yaml(self):
        self.export_yaml()
        messagebox.showinfo("YAML Exported", "annotations.yaml has been saved!")

if __name__ == '__main__':
    root = tk.Tk()
    app = AutoAnnotationApp(root)
    root.mainloop()
