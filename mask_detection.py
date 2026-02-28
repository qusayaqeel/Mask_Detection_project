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

# Optimize model for speed on CPU
model.conf = 0.4    # Confidence threshold (higher = faster)
model.iou = 0.45    # NMS IoU threshold

camera_running = False

# Colors for each class (BGR)
CLASS_COLORS = {
    'with_mask': (0, 180, 0),          # Green
    'without_mask': (0, 0, 220),       # Red
    'mask_weared_incorrect': (0, 140, 255),  # Orange
}

def draw_detections(frame, df):
    """Draw clean bounding boxes with small labels on the frame."""
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_name = row['name']
        conf = float(row['confidence'])
        color = CLASS_COLORS.get(class_name, (200, 200, 200))

        # Draw thin bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Short label
        short_name = class_name.replace('mask_weared_incorrect', 'incorrect').replace('without_mask', 'no mask').replace('with_mask', 'mask')
        label = f"{short_name} {conf:.0%}"

        # Small text above the box
        font_scale = 0.25
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Label background
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return frame

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
    
    df = results.pandas().xyxy[0]
    # Custom rendering with clean labels
    im = results.ims[0].copy()  # Original image (RGB)
    im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im_bgr = draw_detections(im_bgr, df)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im_rgb).resize((600, 450))
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
        cam_idx = int(camera_index_var.get())
        print(f"\nOpening camera index: {cam_idx}")
        cap = cv2.VideoCapture(cam_idx)
        # Lower camera resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            return

        print("Camera started! Processing frames...")
        frame_count = 0

        while camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Skip every other frame for speed
            if frame_count % 2 != 0:
                continue

            # Resize frame for faster inference
            h, w = frame.shape[:2]
            scale = 320 / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # YOLOv5 expects RGB image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, size=320)  # Smaller inference size = faster
            
            df = results.pandas().xyxy[0]
            # Custom rendering with clean labels
            frame_drawn = draw_detections(frame.copy(), df)
            frame_drawn_rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame_drawn_rgb).resize((600, 450))
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

# Camera Index Selector
cam_select_frame = ctk.CTkFrame(root, fg_color="transparent")
cam_select_frame.pack(pady=(10, 5))

cam_label = ctk.CTkLabel(cam_select_frame, text="📹 Camera:",
                         font=("Segoe UI", 14, "bold"), text_color="#1f3c88")
cam_label.pack(side="left", padx=(0, 10))

camera_index_var = ctk.StringVar(value="2")
camera_index_menu = ctk.CTkOptionMenu(cam_select_frame,
    values=["0", "1", "2", "3"],
    variable=camera_index_var,
    font=("Segoe UI", 13), width=60, height=35,
    fg_color="#2196F3", button_color="#1e88e5", button_hover_color="#1565c0")
camera_index_menu.pack(side="left", padx=5)

cam_hint = ctk.CTkLabel(cam_select_frame, text="(0=Laptop, 1=DroidCam, 2=OBS Virtual Camera)",
                        font=("Segoe UI", 12), text_color="gray")
cam_hint.pack(side="left", padx=10)

# Buttons
btn_frame = ctk.CTkFrame(root, fg_color="transparent")
btn_frame.pack(pady=15)

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
