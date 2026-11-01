import io
import time
import joblib
import requests
import numpy as np
import librosa
import streamlit as st
import streamlit.components.v1 as components

# =====================================================================
# 1. KONFIGURASI HALAMAN & GAYA UI (RESPONSIF MOBILE & PC)
# =====================================================================
st.set_page_config(
    page_title="Sistem Deteksi Tangisan Bayi - SVM RBF",
    page_icon="👶",
    layout="centered"
)

st.markdown("""
<style>
    .stApp { background-color: #121417; color: #e4e6eb; }
    .status-card {
        padding: 18px; border-radius: 10px; margin-bottom: 15px;
        border: 1px solid #2d323b; background-color: #1c1f26;
    }
    .alert-crying {
        padding: 20px; border-radius: 10px; background-color: #3b181e;
        border-left: 6px solid #dc2626; color: #fca5a5; font-weight: 600;
    }
    .alert-safe {
        padding: 20px; border-radius: 10px; background-color: #143323;
        border-left: 6px solid #16a34a; color: #86efac; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. LOAD MODEL SVM & SCALER
# =====================================================================
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_CHUNK = SAMPLE_RATE * DURATION
COOLDOWN_WA_DETIK = 30  # Jeda minimal antar notifikasi WA (anti-spam)

@st.cache_resource
def load_ml_artifacts():
    model = joblib.load("model_svm_terbaik.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

try:
    model_svm, scaler_audio = load_ml_artifacts()
except Exception as e:
    st.error(f"Gagal memuat file model/scaler (.pkl). Pastikan 'model_svm_terbaik.pkl' dan 'scaler_audio.pkl' ada di repo. Detail: {e}")
    st.stop()

# =====================================================================
# 3. FUNGSI EKSTRAKSI 182 FITUR AKUSTIK (IDENTIK DENGAN NOTEBOOK)
# =====================================================================
def extract_182_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mengekstrak 182 fitur (Mean & Std dari 40 MFCC, 40 Delta MFCC, Centroid, Rolloff, 7 Contrast, ZCR, RMS)."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) >= sr * 1:
        y = y_trimmed

    if len(y) < SAMPLES_PER_CHUNK:
        y = np.pad(y, (0, SAMPLES_PER_CHUNK - len(y)), mode="constant")
    else:
        y = y[:SAMPLES_PER_CHUNK]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)

    all_features = np.vstack([mfcc, mfcc_delta, centroid, rolloff, contrast, zcr, rms])
    feat_mean = np.mean(all_features, axis=1)
    feat_std = np.std(all_features, axis=1)

    return np.hstack([feat_mean, feat_std]).reshape(1, -1)

# =====================================================================
# 4. FUNGSI PENGIRIMAN NOTIFIKASI WHATSAPP DENGAN COOLDOWN 30 DETIK
# =====================================================================
def kirim_notifikasi_wa(nomor_wa: str, confidence: float, metode: str, url_webhook: str, secret_key: str, token_fonnte: str):
    """Mengirim notifikasi ke WhatsApp via Webhook Bot Panel atau Fonnte dengan batas cooldown 30 detik."""
    waktu_sekarang = time.time()
    selisih_waktu = waktu_sekarang - st.session_state.last_wa_sent_time

    if selisih_waktu < COOLDOWN_WA_DETIK:
        sisa_cooldown = int(COOLDOWN_WA_DETIK - selisih_waktu)
        return False, f"Ditahan oleh Cooldown Anti-Spam ({sisa_cooldown} detik tersisa sebelum notif WA berikutnya)."

    pesan = (
        f"🚨 *PERINGATAN DETEKSI TANGISAN BAYI* 🚨\n\n"
        f"Sistem mendeteksi suara bayi sedang menangis!\n"
        f"• *Status:* Bayi Menangis (Kelas 1)\n"
        f"• *Model:* SVM (RBF Kernel)\n"
        f"• *Tingkat Keyakinan:* {confidence:.2f}%\n"
        f"• *Waktu Deteksi:* {time.strftime('%H:%M:%S WIB')}\n\n"
        f"Mohon segera periksa kondisi bayi Anda."
    )

    try:
        if metode == "Opsi A: Webhook Bot WA Panel (Public URL)":
            if not url_webhook:
                return False, "URL Webhook Panel belum diisi di pengaturan."
            resp = requests.post(
                url_webhook,
                json={"target": nomor_wa, "confidence": round(confidence, 2), "secret_key": secret_key},
                timeout=10
            )
            if resp.status_code == 200:
                st.session_state.last_wa_sent_time = waktu_sekarang
                return True, f"Notifikasi WA berhasil dikirim ke {nomor_wa} (via Panel Bot)."
            return False, f"Gagal kirim ke Panel Bot (HTTP {resp.status_code}): {resp.text}"

        else:  # Opsi B: API Fonnte
            if not token_fonnte:
                return False, "Token API Fonnte belum diisi di pengaturan."
            resp = requests.post(
                "https://api.fonnte.com/send",
                headers={"Authorization": token_fonnte},
                data={"target": nomor_wa, "message": pesan},
                timeout=10
            )
            res_json = resp.json()
            if resp.status_code == 200 and res_json.get("status"):
                st.session_state.last_wa_sent_time = waktu_sekarang
                return True, f"Notifikasi WA berhasil dikirim ke {nomor_wa} (via Fonnte)."
            return False, f"Gagal kirim via Fonnte: {res_json}"

    except Exception as e:
        return False, f"Kendala jaringan saat mengirim WA: {e}"

# =====================================================================
# 5. INISIALISASI STATE APLIKASI
# =====================================================================
if "mic_active" not in st.session_state:
    st.session_state.mic_active = False
if "last_wa_sent_time" not in st.session_state:
    st.session_state.last_wa_sent_time = 0.0
if "riwayat_deteksi" not in st.session_state:
    st.session_state.riwayat_deteksi = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

# =====================================================================
# 6. SIDEBAR PENGATURAN GATEWAY WHATSAPP
# =====================================================================
with st.sidebar:
    st.header("⚙️ Konfigurasi Gateway WA")
    metode_wa = st.radio(
        "Pilih Jalur Pengiriman WA:",
        ["Opsi A: Webhook Bot WA Panel (Public URL)", "Opsi B: API Gateway (Fonnte)"]
    )
    if metode_wa == "Opsi A: Webhook Bot WA Panel (Public URL)":
        url_webhook = st.text_input(
            "Endpoint Public URL Panel:",
            value=st.secrets.get("WEBHOOK_URL", "https://notifbayi.domainpanel.com/kirim-notif")
        )
        secret_key = st.text_input(
            "Secret Key Webhook:",
            value=st.secrets.get("WEBHOOK_SECRET", "KUNCI_RAHASIA_PROYEK"),
            type="password"
        )
        token_fonnte = ""
    else:
        token_fonnte = st.text_input(
            "Token API Fonnte:",
            value=st.secrets.get("FONNTE_TOKEN", ""),
            type="password"
        )
        url_webhook = ""
        secret_key = ""

    st.caption("💡 Tip: Anda juga dapat menyimpan token/URL di menu *Secrets* Streamlit Cloud agar otomatis terisi.")

# =====================================================================
# 7. ANTARMUKA UTAMA (INPUT NOMOR WA & KONTROL MIKROFON)
# =====================================================================
st.title("👶 Deteksi Suara Tangisan Bayi Real-Time")
st.write("Sistem Pemantauan Pintar Berbasis **SVM (RBF Kernel)** dengan Integrasi Peringatan **WhatsApp**.")

# Input Nomor WhatsApp Tujuan
nomor_wa_raw = st.text_input(
    "📱 Masukkan Nomor WhatsApp Tujuan Notifikasi (Wajib diisi sebelum menyalakan mikrofon):",
    placeholder="Contoh: 081234567890 atau 6281234567890",
    disabled=st.session_state.mic_active
)

# Normalisasi dan validasi nomor WA (mengubah awalan 08 menjadi 628)
nomor_wa_clean = "".join(filter(str.isdigit, nomor_wa_raw))
if nomor_wa_clean.startswith("0"):
    nomor_wa_clean = "62" + nomor_wa_clean[1:]
nomor_wa_valid = len(nomor_wa_clean) >= 10

if not nomor_wa_valid:
    st.warning("⚠️ Silakan masukkan nomor WhatsApp yang valid (minimal 10 digit) untuk membuka kunci mikrofon.")

# Tombol Saklar Hidup/Mati Mikrofon (Sesuai aturan: Tidak bisa hidup tanpa nomor WA, tidak mati sebelum dimatikan)
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button(
        "🎙️ Hidupkan Pemantauan Mikrofon",
        disabled=(not nomor_wa_valid or st.session_state.mic_active),
        use_container_width=True,
        type="primary"
    ):
        st.session_state.mic_active = True
        st.rerun()

with col_btn2:
    if st.button(
        "⏹️ Matikan Mikrofon",
        disabled=(not st.session_state.mic_active),
        use_container_width=True
    ):
        st.session_state.mic_active = False
        st.session_state.last_audio_hash = None
        st.rerun()

st.divider()

# =====================================================================
# 8. PEREKAMAN MIKROFON & KLASIFIKASI REAL-TIME
# =====================================================================
if not st.session_state.mic_active:
    st.info("🔒 **Status Mikrofon: NONAKTIF.** Masukkan nomor WhatsApp dan klik **Hidupkan Pemantauan Mikrofon** untuk memulai deteksi suara.")
else:
    st.success(f"🟢 **Status Mikrofon: AKTIF MEMANTAU** | Target Notifikasi WA: `{nomor_wa_clean}` (Cooldown: {COOLDOWN_WA_DETIK} detik)")

    mode_input = st.tabs(["🎙️ Mikrofon Real-Time (Otomatis Per 5 Detik / Manual)", "📂 Uji Upload File Audio (.wav/.mp3)"])

    with mode_input[0]:
        auto_loop = st.checkbox(
            "🔄 Aktifkan Siklus Rekam Otomatis Berkelanjutan (Otomatis memotong & mengirim segmen tiap 5 detik selama mikrofon hidup)",
            value=True
        )

        audio_buffer = st.audio_input("Perekam Suara Aktif (Berfungsi di Browser HP & PC):")

        # Script JS opsional untuk menjaga siklus rekam 5 detik berjalan terus-menerus tanpa perlu klik ulang
        if auto_loop:
            components.html(
                """
                <script>
                const doc = window.parent.document;
                setInterval(() => {
                    const buttons = doc.querySelectorAll('button[aria-label*="Record"], button[aria-label*="Stop"], button[data-testid="stAudioInputActionButton"]');
                    buttons.forEach(btn => {
                        const label = (btn.getAttribute('aria-label') || '').toLowerCase();
                        if (label.includes('record') || label.includes('stop')) {
                            btn.click();
                        }
                    });
                }, 5500);
                </script>
                """,
                height=0,
            )

        if audio_buffer is not None:
            raw_bytes = audio_buffer.getvalue()
            current_hash = hash(raw_bytes)

            if current_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = current_hash
                with st.spinner("Mengekstrak 182 fitur akustik & menjalankan klasifikasi SVM..."):
                    try:
                        y_audio, sr_audio = librosa.load(io.BytesIO(raw_bytes), sr=SAMPLE_RATE, mono=True)
                        fitur_182 = extract_182_features(y_audio, sr_audio)
                        fitur_scaled = scaler_audio.transform(fitur_182)

                        prediksi = int(model_svm.predict(fitur_scaled)[0])
                        probabilitas = model_svm.predict_proba(fitur_scaled)[0]
                        conf_nangis = probabilitas[1] * 100
                        conf_bukan = probabilitas[0] * 100
                        waktu_str = time.strftime("%H:%M:%S")

                        if prediksi == 1:
                            st.markdown(
                                f"<div class='alert-crying'>🚨 [REAL-TIME {waktu_str}] TERDETEKSI BAYI MENANGIS! "
                                f"(Keyakinan Model: {conf_nangis:.2f}%)</div>",
                                unsafe_allow_html=True
                            )
                            terkirim, info_wa = kirim_notifikasi_wa(
                                nomor_wa_clean, conf_nangis, metode_wa, url_webhook, secret_key, token_fonnte
                            )
                            if terkirim:
                                st.toast(info_wa, icon="✅")
                                st.success(f"📨 **Status WhatsApp:** {info_wa}")
                            else:
                                st.warning(f"⏳ **Status WhatsApp:** {info_wa}")
                        else:
                            st.markdown(
                                f"<div class='alert-safe'>✅ [REAL-TIME {waktu_str}] KONDISI AMAN / BUKAN TANGISAN "
                                f"(Keyakinan Bukan Nangis: {conf_bukan:.2f}%)</div>",
                                unsafe_allow_html=True
                            )
                            info_wa = "Tidak dikirim (Suara bukan tangisan)"

                        st.session_state.riwayat_deteksi.insert(0, {
                            "Waktu": waktu_str,
                            "Hasil Klasifikasi": "🚨 Bayi Menangis (1)" if prediksi == 1 else "✅ Bukan Nangis (0)",
                            "Probabilitas Nangis": f"{conf_nangis:.2f}%",
                            "Status Notifikasi WA": info_wa
                        })
                    except Exception as err:
                        st.error(f"Gagal memproses sinyal audio: {err}")

    with mode_input[1]:
        uploaded_file = st.file_uploader("Unggah file rekaman (.wav atau .mp3) untuk pengujian:", type=["wav", "mp3"])
        if uploaded_file is not None:
            st.audio(uploaded_file)
            if st.button("🔍 Analisis File Audio & Kirim Notif Bila Menangis"):
                with st.spinner("Mengekstrak 182 fitur akustik..."):
                    y_up, sr_up = librosa.load(uploaded_file, sr=SAMPLE_RATE, mono=True)
                    fitur_182 = extract_182_features(y_up, sr_up)
                    fitur_scaled = scaler_audio.transform(fitur_182)

                    prediksi = int(model_svm.predict(fitur_scaled)[0])
                    probabilitas = model_svm.predict_proba(fitur_scaled)[0]
                    conf_nangis = probabilitas[1] * 100
                    conf_bukan = probabilitas[0] * 100
                    waktu_str = time.strftime("%H:%M:%S")

                    if prediksi == 1:
                        st.markdown(
                            f"<div class='alert-crying'>🚨 TERDETEKSI BAYI MENANGIS! (Confidence: {conf_nangis:.2f}%)</div>",
                            unsafe_allow_html=True
                        )
                        terkirim, info_wa = kirim_notifikasi_wa(
                            nomor_wa_clean, conf_nangis, metode_wa, url_webhook, secret_key, token_fonnte
                        )
                        if terkirim:
                            st.success(f"📨 {info_wa}")
                        else:
                            st.warning(f"⏳ {info_wa}")
                    else:
                        st.markdown(
                            f"<div class='alert-safe'>✅ BUKAN TANGISAN BAYI (Confidence Aman: {conf_bukan:.2f}%)</div>",
                            unsafe_allow_html=True
                        )
                        info_wa = "Tidak dikirim (Suara bukan tangisan)"

                    st.session_state.riwayat_deteksi.insert(0, {
                        "Waktu": waktu_str,
                        "Hasil Klasifikasi": "🚨 Bayi Menangis (1)" if prediksi == 1 else "✅ Bukan Nangis (0)",
                        "Probabilitas Nangis": f"{conf_nangis:.2f}%",
                        "Status Notifikasi WA": info_wa
                    })

# =====================================================================
# 9. LOG RIWAYAT DETEKSI REAL-TIME DI LAYAR PERANGKAT LUNAK
# =====================================================================
if st.session_state.riwayat_deteksi:
    st.subheader("📊 Log Klasifikasi & Notifikasi Real-Time")
    st.dataframe(st.session_state.riwayat_deteksi[:15], use_container_width=True)