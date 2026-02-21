import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from PIL import Image
import cv2
import threading
import os

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# -------------------------------
# Load the model
# Use local copy of yolov5 instead of downloading from the internet
yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5-master')
model = torch.hub.load(yolov5_path, 'custom', path='mask_detection_yolov5.pt', source='local')

camera_running = False

# -------------------------------
# GUI Functions
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        run_detection(file_path)

def run_detection(file_path):
    results = model(file_path)
    
    im = results.render()[0]
    im_pil = Image.fromarray(im).resize((600, 450)) # larger image for modern look
    imgtk = ctk.CTkImage(light_image=im_pil, dark_image=im_pil, size=(600, 450))

    panel.configure(image=imgtk, text="")
    panel.image = imgtk  # type: ignore

    df = results.pandas().xyxy[0]
    detected = []
    for _, row in df.iterrows():
        class_name = row['name']
        confidence = float(row['confidence']) * 100
        detected.append(f"• {class_name} ({confidence:.2f}%)")

    results_box.configure(state="normal")
    results_box.delete("1.0", "end")
    if detected:
        results_box.insert("end", "\n".join(detected))
    else:
        results_box.insert("end", "⚠️ No face detected")
    results_box.configure(state="disabled")

def open_camera():
    def camera_loop():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            return

        while camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv5 expects RGB image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            
            im = results.render()[0]
            im_pil = Image.fromarray(im).resize((600, 450))
            imgtk = ctk.CTkImage(light_image=im_pil, dark_image=im_pil, size=(600, 450))

            panel.configure(image=imgtk, text="")
            panel.image = imgtk  # type: ignore

            df = results.pandas().xyxy[0]
            detected = []
            for _, row in df.iterrows():
                class_name = row['name']
                confidence = float(row['confidence']) * 100
                detected.append(f"• {class_name} ({confidence:.2f}%)")

            results_box.configure(state="normal")
            results_box.delete("1.0", "end")
            if detected:
                results_box.insert("end", "\n".join(detected))
            else:
                results_box.insert("end", "⚠️ No face detected")
            results_box.configure(state="disabled")

        cap.release()

    global camera_running
    camera_running = True
    t = threading.Thread(target=camera_loop)
    t.daemon = True
    t.start()

def stop_camera():
    global camera_running
    camera_running = False

# -------------------------------
# Tkinter Interface
root = ctk.CTk()
root.title("😷 Face Mask Detection System")
root.geometry("1000x700")

# Title
title = ctk.CTkLabel(root, text="😷 Face Mask Detection System",
                 font=("Segoe UI", 28, "bold"), text_color="#1f3c88")
title.pack(pady=20)

# Frame for Image + Results
content_frame = ctk.CTkFrame(root, fg_color="transparent")
content_frame.pack(pady=10, padx=20, fill="both", expand=True)

# Image Area
panel = ctk.CTkLabel(content_frame, text="Upload an Image or Open Camera", font=("Segoe UI", 16), width=600, height=450, fg_color=("gray85", "gray25"), corner_radius=10)
panel.grid(row=0, column=0, padx=20, pady=10)

# Results Area
results_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
results_frame.grid(row=0, column=1, sticky="n", padx=10, pady=10)

results_title = ctk.CTkLabel(results_frame, text="📊 Detection Results",
                         font=("Segoe UI", 20, "bold"), text_color="#1f3c88")
results_title.pack(anchor="n", pady=(0, 10))

results_box = ctk.CTkTextbox(results_frame, font=("Segoe UI", 16), width=300, height=400,
                      fg_color=("white", "gray20"), text_color=("black", "white"), corner_radius=10)
results_box.pack(pady=10)
results_box.configure(state="disabled")

# Buttons
btn_frame = ctk.CTkFrame(root, fg_color="transparent")
btn_frame.pack(pady=30)

upload_btn = ctk.CTkButton(btn_frame, text="📂 Upload Image", font=("Segoe UI", 16, "bold"),
                       command=upload_image, fg_color="#4CAF50", hover_color="#45a049", width=180, height=45, corner_radius=8)
upload_btn.pack(side="left", padx=15)

camera_btn = ctk.CTkButton(btn_frame, text="🎥 Open Camera", font=("Segoe UI", 16, "bold"),
                       command=open_camera, fg_color="#2196F3", hover_color="#1e88e5", width=180, height=45, corner_radius=8)
camera_btn.pack(side="left", padx=15)

stop_btn = ctk.CTkButton(btn_frame, text="⏹ Stop Camera", font=("Segoe UI", 16, "bold"),
                     command=stop_camera, fg_color="#FF9800", hover_color="#fb8c00", width=180, height=45, corner_radius=8)
stop_btn.pack(side="left", padx=15)

exit_btn = ctk.CTkButton(btn_frame, text="❌ Exit", font=("Segoe UI", 16, "bold"),
                     command=root.quit, fg_color="#f44336", hover_color="#e53935", width=180, height=45, corner_radius=8)
exit_btn.pack(side="left", padx=15)

root.mainloop()
