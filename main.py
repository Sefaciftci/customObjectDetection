import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
from ultralytics import YOLO
import yaml

import os
import sys

def get_data_file_path(file_name):
    if getattr(sys, 'frozen', False):
        # PyInstaller kullanarak çalıştırıldığında
        base_path = sys._MEIPASS
    else:
        # Geliştirme sırasında
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, file_name)

yaml_path = get_data_file_path('ultralytics/cfg/default.yaml')


def get_config_path():
    if getattr(sys, 'frozen', False):
        # PyInstaller'dan çalışıyorsa
        return os.path.join(sys._MEIPASS, 'config', 'default.yaml')
    else:
        # Doğrudan çalışıyorsa
        return 'config/default.yaml'

config_path = get_config_path()
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# Modeli yükle
model = YOLO('weights/best.pt')


def load_image():
    # Fotoğraf yükleme diyalog kutusu
    filepath = filedialog.askopenfilename()

    # Fotoğrafı oku ve GUI'de göster
    img = Image.open(filepath)
    img = img.resize((350, 350))
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img

    # Modeli çalıştır ve sonucu al
    results = model(filepath)
    display_results(results)


def display_results(results):
    # Sonuçları GUI'de göster
    result_text = "Sonuçlar:\n"

    # Sonuçları kontrol et ve ekrana yazdır
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = results[0].names[cls]
            result_text += f"Etiket: {label}, Dogruluk: {conf:.2f}\n"

    result_label.config(text=result_text)


# GUI penceresini oluştur
root = tk.Tk()
root.title("YOLOv8 Nesne Tanıma")

# Fotoğraf gösterim etiketi
label = Label(root)
label.pack()

# Sonuç etiketini oluştur
result_label = Label(root, text="Sonuçlar:", justify=tk.LEFT)
result_label.pack()

# Fotoğraf yükleme düğmesi
button = tk.Button(root, text="Fotoğraf Yükle", command=load_image)
button.pack()

root.mainloop()