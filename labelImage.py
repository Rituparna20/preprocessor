import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageLabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling Tool")

        self.image_index = 0
        self.image_list = []
        self.labels = []
        self.current_image_path = None

        self.label_options = ["Cat", "Dog", "Other"]  # Add more labels as needed
        self.output_file = "labels.csv"

        self.setup_ui()

    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        for label in self.label_options:
            btn = tk.Button(btn_frame, text=label, command=lambda l=label: self.save_label(l))
            btn.pack(side=tk.LEFT, padx=5)

        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=5)

        tk.Button(nav_frame, text="Load Images", command=self.load_images).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)

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
            messagebox.showerror("Error", "No images found in folder!")
            return

        self.image_index = 0
        self.labels = []
        self.show_image()

    def show_image(self):
        if not self.image_list:
            return

        self.current_image_path = self.image_list[self.image_index]
        img = Image.open(self.current_image_path)
        img.thumbnail((500, 500))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(250, 250, image=self.tk_img)

    def save_label(self, label):
        if not self.current_image_path:
            return

        image_name = os.path.basename(self.current_image_path)
        self.labels.append((image_name, label))

        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([image_name, label])

        self.next_image()

    def next_image(self):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.show_image()
        else:
            messagebox.showinfo("Done", "No more images to label!")

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.show_image()

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageLabelApp(root)
    root.mainloop()
