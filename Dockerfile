# Gunakan base image Python 3.11-slim agar identik dengan environment training laptop (Python 3.11)
FROM python:3.11-slim

# Hindari pembuatan file .pyc dan pastikan log langsung keluar ke terminal
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Jakarta

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Install dependensi sistem operasi untuk pemrosesan audio (libsndfile & ffmpeg) serta curl untuk healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt terlebih dahulu agar caching layer Docker optimal
COPY requirements.txt .

# Install dependensi Python sesuai versi yang telah disamakan dengan laptop
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek ke dalam container
COPY . .

# Expose port aplikasi (Port yang ditentukan: 8022)
EXPOSE 8022

# Healthcheck untuk memonitor kesiapan aplikasi
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Jalankan Streamlit dengan bind ke 0.0.0.0 dan nonaktifkan proteksi CORS/XSRF agar dapat diakses dari IP jaringan server lab
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
