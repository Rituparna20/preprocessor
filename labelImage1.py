import os
import yaml
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

class AutoAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Image Annotation Tool")

        self.image_index = 0
        self.image_list = []
        self.annotations = []
        self.current_image_path = None
        self.model = YOLO("yolov8n.pt")  # Using small YOLOv8 model
        self.current_bbox = None
        self.current_label = ""

        self.setup_ui()

    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        label_frame = tk.Frame(self.root)
        label_frame.pack(pady=5)

        tk.Label(label_frame, text="Label: ").pack(side=tk.LEFT)
        self.label_entry = tk.Entry(label_frame)
        self.label_entry.pack(side=tk.LEFT)

        tk.Button(label_frame, text="Save Label", command=self.save_label).pack(side=tk.LEFT, padx=5)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5)

        tk.Button(control_frame, text="Load Images", command=self.load_images).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Export YAML", command=self.export_yaml).pack(side=tk.LEFT, padx=5)

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
        self.show_image()

    def show_image(self):
        if not self.image_list:
            return

        self.current_image_path = self.image_list[self.image_index]
        img = cv2.imread(self.current_image_path)
        results = self.model(img)[0]

        max_area = 0
        best_box = None
        best_cls = None

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)
                best_cls = int(box.cls[0].item())

        if best_box:
            self.current_bbox = best_box
            self.current_label = self.model.names[best_cls]
            self.label_entry.delete(0, tk.END)
            self.label_entry.insert(0, self.current_label)

            img = cv2.rectangle(img, best_box[:2], best_box[2:], (0, 255, 0), 2)
            img = cv2.putText(img, self.current_label, (best_box[0], best_box[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_pil.thumbnail((640, 480))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(320, 240, image=self.tk_img)

    def save_label(self):
        label = self.label_entry.get().strip()
        if not label or not self.current_bbox:
            return

        self.annotations.append({
            "image": os.path.basename(self.current_image_path),
            "label": label,
            "bbox": list(map(int, self.current_bbox))
        })
        self.next_image()

    def next_image(self):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.show_image()
        else:
            messagebox.showinfo("Done", "All images processed!")

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.show_image()

    def export_yaml(self):
        data = {"annotations": self.annotations}
        with open("annotations.yaml", "w") as f:
            yaml.dump(data, f)
        messagebox.showinfo("Exported", "Annotations saved to annotations.yaml")

if __name__ == '__main__':
    root = tk.Tk()
    app = AutoAnnotationApp(root)
    root.mainloop()
