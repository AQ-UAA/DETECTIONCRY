# 👶 Baby Monitor - Deteksi Tangisan Bayi (Klasifikasi Sinyal Audio)

Proyek klasifikasi suara berbasis Machine Learning untuk mendeteksi tangisan bayi secara real-time dan melalui unggahan file audio, dilengkapi sistem notifikasi peringatan ke WhatsApp.

Dibuat untuk memenuhi tugas mata kuliah **Pengolahan Sinyal Digital (PSD)**.

---

## 📌 Ringkasan Proyek

- **Model Machine Learning:** Support Vector Machine (SVM) dengan RBF Kernel.
- **Ekstraksi Fitur (182 Fitur):**
  - MFCC (40 koefisien) + Delta MFCC
  - Spectral Centroid
  - Spectral Rolloff
  - Spectral Contrast
  - Zero Crossing Rate (ZCR)
  - Root Mean Square Energy (RMS)
  - Agregasi statistik: *Mean* & *Standard Deviation*.
- **Antarmuka Pengguna:** Streamlit Web UI (Tema WhatsApp UI, mobile friendly).
- **Fitur Pengujian:**
  1. **Mikrofon Real-time:** Merekam dan menganalisis potongan audio per segmen.
  2. **Uji File Audio (.wav / .mp3):** Analisis langsung file rekaman.
  3. **Notifikasi WhatsApp:** Mengirim pesan peringatan otomatis saat tangisan terdeteksi (via Webhook / Fonnte).

---

## 📦 Model & Dataset

- Model ML (`model_svm_terbaik.pkl` ~ 313 KB) dan StandardScaler (`scaler_audio.pkl` ~ 7 KB) **sudah disertakan langsung di root repository**.
- Karena ukuran model sangat ringan (< 1 MB), model dapat langsung digunakan setelah clone tanpa perlu mengunduh file eksternal atau konfigurasi Git LFS.

---

## 🚀 Panduan Menjalankan dengan Docker

Proyek ini telah dikonfigurasi untuk berjalan dengan mudah di server laboratorium (Proxmox) maupun komputer lokal hanya dengan satu perintah:

### 1. Prasyarat
- Docker & Docker Compose sudah terpasang.

### 2. Menjalankan Aplikasi
Buka terminal di root direktori proyek, lalu jalankan:

```bash
docker compose up --build
```

*(Opsi background mode / daemon: `docker compose up --build -d`)*

Docker akan secara otomatis:
1. Menginstal dependensi sistem operasi (`libsndfile1`, `ffmpeg`, `curl`).
2. Menginstal library Python yang terdaftar di `requirements.txt`.
3. Menjalankan server Streamlit pada port **8022**.

### 3. Akses Aplikasi
Buka peramban (browser) dan akses:
- **Lokal:** `http://localhost:8022`
- **Server Lab / Proxmox:** `http://<IP-SERVER-PROXMOX>:8022`

### 4. Menghentikan Aplikasi
Untuk mematikan container:
```bash
docker compose down
```

---

## 🌐 Informasi Port

- **Port yang dibuka:** `8022` (1 port).

---

## 💡 Catatan Pengujian Audio & Mikrofon

- **Akses Mikrofon Browser:** Menurut standar keamanan peramban modern (Chrome/Firefox/Edge), akses mikrofon streaming (`getUserMedia`) hanya diizinkan pada *Secure Context* (`https://` atau `http://localhost`).
- **Pengujian di Server Lab via IP HTTP:** Jika server lab diakses melalui IP biasa tanpa HTTPS (contoh: `http://192.168.x.x:8022`), browser klien mungkin membatasi mikrofon langsung. Oleh karena itu, gunakan fitur **"📂 Uji dengan file audio (.wav / .mp3)"** yang tersedia di bagian bawah aplikasi untuk menguji klasifikasi suara dengan file rekaman.

---

## 📂 Struktur File Repository

```text
├── Dockerfile              # Konfigurasi build image container
├── docker-compose.yml      # Konfigurasi deployment service Docker
├── requirements.txt        # Daftar dependensi library Python
├── .dockerignore           # File pengabaian untuk Docker build
├── app.py                  # Skrip utama aplikasi Streamlit
├── model_svm_terbaik.pkl   # Model klasifikasi SVM
├── scaler_audio.pkl        # Scaler fitur audio
├── packages.txt            # Daftar paket sistem Linux
├── mic_component/          # Komponen kustom perekam mikrofon web
│   └── index.html
└── README.md               # Dokumentasi dan panduan proyek
```
