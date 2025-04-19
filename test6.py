import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import pyttsx3
import time
from PIL import Image, ImageTk

class ModernStyle:
    # Modern dark theme colors
    BG_DARK = "#1E1E1E"      # Dark background
    BG_LIGHT = "#2D2D2D"     # Lighter background
    PRIMARY = "#BB86FC"      # Purple accent
    SECONDARY = "#03DAC6"    # Teal accent
    TEXT_PRIMARY = "#FFFFFF" # White text
    TEXT_SECONDARY = "#B3B3B3" # Gray text
    ERROR = "#CF6679"        # Error red
    SUCCESS = "#03DAC6"      # Success teal
    
    @classmethod
    def apply_modern_style(cls):
        style = ttk.Style()
        
        style.theme_create("ModernDark", parent="alt", settings={
            "TFrame": {
                "configure": {"background": cls.BG_DARK}
            },
            "TLabel": {
                "configure": {
                    "background": cls.BG_DARK,
                    "foreground": cls.TEXT_PRIMARY,
                    "padding": 5,
                    "font": ("Segoe UI", 10)
                }
            },
            "TLabelframe": {
                "configure": {
                    "background": cls.BG_DARK,
                    "foreground": cls.TEXT_PRIMARY,
                    "padding": 10,
                    "bordercolor": cls.PRIMARY,
                    "relief": "solid"
                }
            },
            "TLabelframe.Label": {
                "configure": {
                    "background": cls.BG_DARK,
                    "foreground": cls.PRIMARY,
                    "font": ("Segoe UI", 11, "bold")
                }
            },
            "TButton": {
                "configure": {
                    "background": cls.PRIMARY,
                    "foreground": cls.TEXT_PRIMARY,
                    "padding": (20, 10),
                    "font": ("Segoe UI", 10, "bold"),
                    "borderwidth": 0
                },
                "map": {
                    "background": [("active", cls.SECONDARY)],
                    "foreground": [("active", cls.BG_DARK)]
                }
            },
            "Horizontal.TScale": {
                "configure": {
                    "background": cls.BG_DARK,
                    "troughcolor": cls.BG_LIGHT,
                    "sliderlength": 20,
                    "sliderrelief": "flat"
                }
            }
        })
        
        style.theme_use("ModernDark")
        return style

class SignLanguageDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection System")
        self.root.state('zoomed')
        
        # Apply modern dark theme
        self.style = ModernStyle.apply_modern_style()
        
        # Configure root window
        self.root.configure(bg=ModernStyle.BG_DARK)
        
        # Initialize components
        self.setup_variables()
        self.create_modern_gui()
        self.initialize_detection_system()
        
    def setup_variables(self):
        self.is_running = False
        self.camera_active = False
        self.last_prediction = None
        self.last_prediction_time = 0
        self.last_added_word = None
        self.prediction_threshold = 0.8
        self.cooldown_time = 2.0
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
    def create_modern_gui(self):
        # Create main container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_label = ttk.Label(
            header_frame,
            text="Sign Language Detection System",
            font=("Segoe UI", 24, "bold"),
            foreground=ModernStyle.PRIMARY
        )
        header_label.pack()
        
        # Create content area
        content = ttk.Frame(main_container)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Create control and video panels
        self.create_modern_control_panel(content)
        self.create_modern_video_panel(content)
        
    def create_modern_control_panel(self, parent):
        control_frame = ttk.Frame(parent, width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Status section
        status_frame = self.create_section(control_frame, "System Status")
        self.status_label = ttk.Label(
            status_frame,
            text="System Ready",
            font=("Segoe UI", 12)
        )
        self.status_label.pack(pady=10)
        
        # Detection results section
        results_frame = self.create_section(control_frame, "Detection Results")
        self.prediction_label = ttk.Label(
            results_frame,
            text="Waiting for gesture...",
            font=("Segoe UI", 12)
        )
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(
            results_frame,
            text="Confidence: 0%",
            font=("Segoe UI", 12)
        )
        self.confidence_label.pack(pady=5)
        
        # Text output section
        text_frame = self.create_section(control_frame, "Detected Text")
        
        text_style = {
            'bg': ModernStyle.BG_LIGHT,
            'fg': ModernStyle.TEXT_PRIMARY,
            'font': ('Segoe UI', 12),
            'relief': 'flat',
            'padx': 10,
            'pady': 10,
            'insertbackground': ModernStyle.PRIMARY
        }
        
        self.text_output = tk.Text(text_frame, height=8, **text_style)
        self.text_output.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Control buttons section
        controls_frame = self.create_section(control_frame, "Controls")
        
        self.start_button = self.create_button(
            controls_frame, "Start Camera", self.toggle_camera
        )
        self.create_button(controls_frame, "Clear Text", self.clear_text)
        self.create_button(controls_frame, "Read Text", self.read_text)
        self.create_button(
            controls_frame, "Delete Last Word", self.delete_last_word
        )
        
        # Settings section
        settings_frame = self.create_section(control_frame, "Settings")
        ttk.Label(
            settings_frame,
            text="Confidence Threshold:",
            font=("Segoe UI", 10, "bold")
        ).pack(pady=(0, 5))
        
        self.threshold_var = tk.DoubleVar(value=0.8)
        threshold_scale = ttk.Scale(
            settings_frame,
            from_=0.1,
            to=1.0,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL
        )
        threshold_scale.pack(fill=tk.X)
        
    def create_modern_video_panel(self, parent):
        video_frame = self.create_section(parent, "Camera Feed")
        video_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
    def create_section(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=15)
        frame.pack(fill=tk.X, pady=(0, 15))
        return frame
        
    def create_button(self, parent, text, command):
        btn = ttk.Button(parent, text=text, command=command)
        btn.pack(fill=tk.X, pady=5)
        return btn
        
    def initialize_detection_system(self):
        self.cap = None
        self.detector = HandDetector(maxHands=2)
        self.classifier = Classifier("new_model/keras_model.h5", "new_model/labels.txt")
        
        
    def toggle_camera(self):
        if not self.camera_active:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open camera")
                self.camera_active = True
                self.start_button.config(text="Stop Camera")
                self.update_status("Camera Active", False)
                self.update_video()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
                self.update_status("Camera Error", True)
        else:
            self.camera_active = False
            if self.cap:
                self.cap.release()
            self.start_button.config(text="Start Camera")
            self.update_status("Camera Stopped")
            self.video_label.config(image='')
            
    def update_video(self):
        if not self.camera_active:
            return
            
        success, img = self.cap.read()
        if not success:
            self.update_status("Failed to grab frame", True)
            return
            
        hands, img = self.detector.findHands(img)
        
        if hands:
            self.process_hand_detection(hands, img)
        else:
            self.prediction_label.config(text="Waiting for gesture...")
            self.confidence_label.config(text="Confidence: 0%")
            self.last_added_word = None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=img)
        self.video_label.image = img
        
        if self.camera_active:
            self.root.after(10, self.update_video)
            
    def process_hand_detection(self, hands, img):
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255
        
        offset = 20
        y1 = max(y - offset, 0)
        y2 = min(y + h + offset, img.shape[0])
        x1 = max(x - offset, 0)
        x2 = min(x + w + offset, img.shape[1])
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size != 0:
            self.process_hand_image(imgCrop, imgWhite, x1, y1)
            
    def process_hand_image(self, imgCrop, imgWhite, x1, y1):
        aspectRatio = imgCrop.shape[0] / imgCrop.shape[1]
        
        if aspectRatio > 1:
            k = 300 / imgCrop.shape[0]
            wCal = max(int(imgCrop.shape[1] * k), 1)
            imgResize = cv2.resize(imgCrop, (wCal, 300))
            wGap = (300 - wCal) // 2
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = 300 / imgCrop.shape[1]
            hCal = max(int(imgCrop.shape[0] * k), 1)
            imgResize = cv2.resize(imgCrop, (300, hCal))
            hGap = (300 - hCal) // 2
            imgWhite[hGap:hCal + hGap, :] = imgResize
            
        prediction, index = self.classifier.getPrediction(imgWhite)
        
        if 0 <= index < len(self.labels):
            if prediction[index] > self.threshold_var.get():
                self.handle_prediction(self.labels[index], prediction[index])
                
    def handle_prediction(self, predicted_letter, confidence):
        current_time = time.time()
        confidence_pct = int(confidence * 100)
        
        self.prediction_label.config(text=f"Detected: {predicted_letter}")
        self.confidence_label.config(text=f"Confidence: {confidence_pct}%")
        
        if (predicted_letter != self.last_prediction or 
            current_time - self.last_prediction_time > self.cooldown_time):
            
            self.engine.say(f"Letter {predicted_letter}")
            self.engine.runAndWait()
            
            if predicted_letter != self.last_added_word:
                current_text = self.text_output.get("1.0", tk.END).strip()
                if current_text:
                    self.text_output.insert(tk.END, " " + predicted_letter)
                else:
                    self.text_output.insert(tk.END, predicted_letter)
                self.last_added_word = predicted_letter
                
            self.last_prediction = predicted_letter
            self.last_prediction_time = current_time
            
    def clear_text(self):
        self.text_output.delete("1.0", tk.END)
        
    def read_text(self):
        text = self.text_output.get("1.0", tk.END).strip()
        if text:
            self.engine.say(text)
            self.engine.runAndWait()
            
    def delete_last_word(self):
        text = self.text_output.get("1.0", tk.END).strip()
        if text:
            words = text.split()
            if words:
                new_text = ' '.join(words[:-1])
                self.text_output.delete("1.0", tk.END)
                self.text_output.insert("1.0", new_text)
                
    def update_status(self, message, is_error=False):
        color = ModernStyle.ERROR if is_error else ModernStyle.SUCCESS
        self.status_label.configure(text=message, foreground=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageDetector(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (
        app.camera_active and app.toggle_camera(),
        root.destroy()
    ))
    root.mainloop()