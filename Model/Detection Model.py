import cv2
import time
from ultralytics import YOLO
from gtts import gTTS
import io
import pygame

# Inisialisasi model YOLOv8 dengan model hasil pelatihan
model = YOLO('runs/detect/custom_yolov87/weights/best.pt')  # Gantilah dengan path yang sesuai

# Inisialisasi pygame untuk memutar audio
pygame.mixer.init()

def speak(text, lang='id'):
    "Fungsi untuk memberikan umpan balik suara menggunakan gTTS tanpa menyimpan file."
    try:
        # Konversi teks ke audio menggunakan gTTS dan simpan dalam memori
        tts = gTTS(text=text, lang=lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        # Load audio ke pygame mixer dan mainkan
        pygame.mixer.music.load(audio_fp, 'mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Tunggu hingga audio selesai diputar
            time.sleep(0.1)
    except Exception as e:
        print(f"Terjadi kesalahan saat menghasilkan umpan balik suara: {e}")

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Gagal membuka kamera.")
else:
    print("Tekan 'q' untuk keluar.")

# Variabel untuk melacak waktu terakhir umpan balik suara diberikan
last_feedback_time = time.time()  # Gunakan waktu saat ini sebagai titik awal

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break
    
    # Deteksi objek menggunakan YOLO
    results = model(frame)
    
    # Tampilkan hasil deteksi pada frame
    annotated_frame = results[0].plot()
    cv2.imshow("Deteksi Objek TunaNetra", annotated_frame)
    
    frame_height, frame_width, _ = frame.shape
    detected_objects = {}  # Menyimpan jumlah objek yang terdeteksi
    
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.5:  # Ambang batas deteksi
            object_name = model.names[int(class_id)]  # Nama objek
            if object_name not in detected_objects:
                detected_objects[object_name] = 0
            detected_objects[object_name] += 1
    
    # Ambil waktu saat ini
    current_time = time.time()

    # Cek apakah sudah lebih dari 7 detik sejak umpan balik terakhir
    if detected_objects and (current_time - last_feedback_time >= 7):
        # Kumpulkan objek yang terdeteksi 
        feedback = "Didepan ada: "
        for obj, count in detected_objects.items():
            feedback += f"{count} {obj}, "
        feedback = feedback.rstrip(", ")  # Menghapus koma terakhir
        print(feedback)
        speak(feedback)  # Berikan umpan balik suara

        # Update waktu terakhir umpan balik suara diberikan
        last_feedback_time = current_time  # Perbarui waktu terakhir umpan balik suara
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
