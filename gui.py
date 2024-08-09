import tkinter as tk
from tkinter import filedialog, Label, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO('weights/best.pt')


def load_image():
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not filepath:
        return

    img = Image.open(filepath)
    img = img.resize((400, 400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img

    # Model sonuçlarını al
    results = model(filepath)
    display_results(results)


def display_results(results):
    # Sonuçları GUI'de göster
    result_text = "Results:\n"

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = results[0].names[cls]
            result_text += f"Label: {label}, Confidence: {conf:.2f}\n"

    result_label.config(text=result_text)
    result_label.pack(pady=10)


def about():
    messagebox.showinfo("About", "YOLOv8 Object Detection Application\nDeveloped with Tkinter and PyTorch")


# Ana pencereyi oluştur
root = tk.Tk()
root.title("YOLOv8 Object Detection")
root.geometry("1200x800")
root.configure(bg="#2e2e2e")

# Başlık etiketi
title_label = Label(root, text="Object Detection", font=("Arial", 24), bg="#2e2e2e", fg="white")
title_label.pack(pady=20)

# Fotoğraf görüntüleme etiketi
label = Label(root, bg="#2e2e2e")
label.pack()

# Sonuç etiketi
result_label = Label(root, text="", font=("Arial", 14), bg="#2e2e2e", fg="white")
result_label.pack()

# Düğmeler için bir çerçeve oluştur
button_frame = tk.Frame(root, bg="#2e2e2e")
button_frame.pack(pady=20)

# Fotoğraf yükleme düğmesi
upload_button = tk.Button(
    button_frame, text="Upload Image", command=load_image,
    font=("Arial", 14), bg="#007acc", fg="white", padx=10, pady=5
)
upload_button.grid(row=0, column=0, padx=10)

# Hakkında düğmesi
about_button = tk.Button(
    button_frame, text="About", command=about,
    font=("Arial", 14), bg="#007acc", fg="white", padx=10, pady=5
)
about_button.grid(row=0, column=1, padx=10)

# Ana döngüyü başlat
root.mainloop()