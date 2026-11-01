import html
import queue
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import librosa
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

# =====================================================================
# 1. KONFIGURASI
# =====================================================================
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_CHUNK = SAMPLE_RATE * DURATION
COOLDOWN_WA_DETIK = 30     # jeda minimal antar notifikasi WA (anti-spam)
NOTIF_TTL = 8              # lama notifikasi melayang di layar (detik)
SILENCE_RMS = 0.002        # di bawah ini dianggap senyap (tidak dianalisis)
TZ = ZoneInfo("Asia/Jakarta")

st.set_page_config(
    page_title="Baby Monitor - Deteksi Tangisan",
    page_icon="👶",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def jam() -> str:
    return datetime.now(TZ).strftime("%H:%M:%S")


# =====================================================================
# 2. GAYA UI (TEMA CHAT WHATSAPP, MOBILE FRIENDLY)
# =====================================================================
CSS = """
<style>
:root{
  --bg:#0b141a; --panel:#111b21; --bubble:#202c33; --line:#1f2c34;
  --text:#e9edef; --muted:#8696a0; --green:#00a884; --red:#ef4444; --blue:#53bdeb;
}
.stApp{background:var(--bg); color:var(--text);}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
[data-testid="stHeader"]{background:transparent;}
[data-stale="true"]{opacity:1 !important;}
.block-container{padding:.8rem .9rem 4rem !important; max-width:520px;}

/* App bar */
.appbar{display:flex; align-items:center; gap:12px; padding:12px 14px; margin-bottom:12px;
  background:var(--panel); border-radius:16px; border:1px solid var(--line);}
.avatar{width:44px; height:44px; border-radius:50%; display:grid; place-items:center; font-size:24px;
  background:linear-gradient(135deg,#00a884,#0b6e5a);}
.ab-title{font-weight:700; font-size:17px; line-height:1.2;}
.ab-sub{font-size:12.5px; color:var(--muted);}

/* Input & tombol */
.stTextInput input{background:var(--bubble) !important; color:var(--text) !important;
  border:1px solid var(--line) !important; border-radius:12px !important;}
.stTextInput label p{color:var(--muted) !important; font-size:13px;}
.stButton>button{border-radius:12px; font-weight:600; border:1px solid var(--line);
  background:var(--bubble); color:var(--text); min-height:46px;}
.stButton>button:hover{border-color:var(--green); color:var(--green);}
[data-testid="stExpander"]{background:var(--panel); border:1px solid var(--line); border-radius:14px;}

/* Kartu status */
.status{display:flex; align-items:center; gap:14px; padding:16px; margin:10px 0 6px;
  background:var(--panel); border:1px solid var(--line); border-radius:16px;}
.pulse{width:14px; height:14px; border-radius:50%; flex:none; color:var(--green);
  background:currentColor; box-shadow:0 0 0 0 currentColor; animation:pulse 1.6s infinite;}
.status.off .pulse{color:#667781; animation:none;}
.status.alert{border-color:var(--red); background:#2a1317;}
.status.alert .pulse{color:var(--red);}
.st-title{font-weight:700; font-size:15px;}
.st-sub{font-size:12.5px; color:var(--muted); margin-top:2px;}
@keyframes pulse{0%{box-shadow:0 0 0 0 currentColor;} 70%{box-shadow:0 0 0 14px transparent;} 100%{box-shadow:0 0 0 0 transparent;}}
.meter{height:6px; border-radius:99px; background:var(--bubble); overflow:hidden; margin:0 2px 10px;}
.meter>div{height:100%; background:linear-gradient(90deg,#00a884,#53bdeb); border-radius:99px; transition:width .4s;}

/* Statistik */
.stats{display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:12px;}
.stat{background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:10px; text-align:center;}
.stat b{display:block; font-size:20px;}
.stat span{font-size:11.5px; color:var(--muted);}

/* Notifikasi melayang ala WhatsApp (layer paling atas) */
.notif-stack{position:fixed; top:calc(env(safe-area-inset-top,0px) + 10px); left:50%;
  transform:translateX(-50%); width:min(94vw,440px); z-index:2147483000;
  display:flex; flex-direction:column; gap:8px; pointer-events:none;}
.notif{display:flex; gap:12px; align-items:flex-start; padding:12px 14px; border-radius:18px;
  background:rgba(32,44,51,.97); backdrop-filter:blur(12px); border:1px solid #2a3942;
  box-shadow:0 12px 36px rgba(0,0,0,.55); animation:notifLife 8s ease forwards;}
.notif.cry{border-left:5px solid var(--red);}
.notif.ok{border-left:5px solid var(--green);}
.n-ico{width:40px; height:40px; border-radius:50%; display:grid; place-items:center; flex:none;
  font-size:20px; background:linear-gradient(135deg,#00a884,#0b6e5a);}
.notif.cry .n-ico{background:linear-gradient(135deg,#ef4444,#991b1b);}
.n-head{display:flex; justify-content:space-between; font-size:12px; color:var(--muted);}
.n-head b{color:var(--text); font-size:13px;}
.n-title{font-weight:700; font-size:14.5px; margin-top:2px;}
.n-body{font-size:13px; color:#c4cdd2; margin-top:1px;}
@keyframes notifLife{
  0%{transform:translateY(-130%); opacity:0;}
  6%{transform:translateY(0); opacity:1;}
  92%{transform:translateY(0); opacity:1;}
  100%{transform:translateY(-130%); opacity:0;}
}

/* Log gaya chat */
.chat{display:flex; flex-direction:column-reverse; gap:8px; max-height:460px; overflow-y:auto;
  padding:14px 10px; border-radius:16px; border:1px solid var(--line);
  background:
    radial-gradient(circle at 20% 20%, rgba(0,168,132,.06) 0, transparent 40%),
    radial-gradient(circle at 80% 70%, rgba(83,189,235,.05) 0, transparent 40%),
    #0d1a21;}
.bubble{max-width:88%; padding:9px 12px 6px; border-radius:4px 14px 14px 14px; background:var(--bubble);
  align-self:flex-start; box-shadow:0 1px 2px rgba(0,0,0,.4);}
.bubble.cry{border-left:4px solid var(--red);}
.b-who{font-size:12px; font-weight:700; color:var(--green);}
.bubble.cry .b-who{color:#fca5a5;}
.b-txt{font-size:14px; margin-top:2px;}
.b-meta{font-size:11px; color:var(--muted); text-align:right; margin-top:3px;}
.tick{color:#667781; letter-spacing:-3px; margin-left:4px;}
.tick.ok{color:var(--blue);}
.pill{align-self:center; font-size:11.5px; color:var(--muted); background:rgba(32,44,51,.85);
  padding:4px 12px; border-radius:99px;}
.empty{align-self:center; text-align:center; color:var(--muted); font-size:13px; padding:26px 10px;}
.sec-title{font-size:13px; color:var(--muted); margin:14px 2px 6px; font-weight:600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =====================================================================
# 3. LOAD MODEL
# =====================================================================
@st.cache_resource
def load_ml_artifacts():
    return joblib.load("model_svm_terbaik.pkl"), joblib.load("scaler_audio.pkl")


try:
    model_svm, scaler_audio = load_ml_artifacts()
except Exception:
    st.error("Gagal memuat model/scaler (.pkl). Pastikan file ada di repo.")
    st.stop()

# =====================================================================
# 4. EKSTRAKSI 182 FITUR (IDENTIK DENGAN NOTEBOOK)
# =====================================================================
def extract_182_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
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

    allf = np.vstack([mfcc, mfcc_delta, centroid, rolloff, contrast, zcr, rms])
    return np.hstack([allf.mean(axis=1), allf.std(axis=1)]).reshape(1, -1)


def classify(y: np.ndarray, sr: int = SAMPLE_RATE):
    """Return (prediksi, p_nangis, p_bukan)."""
    fitur = scaler_audio.transform(extract_182_features(y, sr))
    pred = int(model_svm.predict(fitur)[0])
    proba = model_svm.predict_proba(fitur)[0]
    return pred, float(proba[1]), float(proba[0])


# =====================================================================
# 5. KLIEN WHATSAPP (KONFIGURASI HANYA DARI st.secrets, TIDAK DITAMPILKAN)
# =====================================================================
def _secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


class WAClient:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_sent = 0.0
        self.method = _secret("WA_METHOD", "webhook").lower()   # "webhook" / "fonnte"
        self.url = _secret("WEBHOOK_URL")
        self.secret = _secret("WEBHOOK_SECRET")
        self.fonnte = _secret("FONNTE_TOKEN")

    @property
    def configured(self) -> bool:
        if self.method == "fonnte":
            return bool(self.fonnte)
        return bool(self.url and self.secret)

    def send(self, nomor: str, confidence: float, is_test: bool = False):
        if not self.configured:
            return False, "Gateway WA belum dikonfigurasi di server."

        with self.lock:  # cooldown hanya untuk deteksi asli
            sisa = COOLDOWN_WA_DETIK - (time.time() - self.last_sent)
            if not is_test and sisa > 0:
                return False, f"Ditahan anti-spam ({int(sisa) + 1} dtk lagi)"
            prev = self.last_sent
            if not is_test:
                self.last_sent = time.time()

        pesan = (
            "🚨 *PERINGATAN DETEKSI TANGISAN BAYI* 🚨\n\n"
            "Sistem mendeteksi suara bayi sedang menangis!\n"
            f"• *Status:* {'UJI COBA KONEKSI (TEST)' if is_test else 'Bayi Menangis'}\n"
            "• *Model:* SVM (RBF Kernel)\n"
            f"• *Tingkat Keyakinan:* {confidence:.2f}%\n"
            f"• *Waktu:* {jam()} WIB\n\n"
            "Mohon segera periksa kondisi bayi Anda."
        )
        try:
            if self.method == "fonnte":
                r = requests.post(
                    "https://api.fonnte.com/send",
                    headers={"Authorization": self.fonnte},
                    data={"target": nomor, "message": pesan},
                    timeout=12,
                )
                ok = r.status_code == 200 and bool(r.json().get("status"))
            else:
                r = requests.post(
                    self.url,
                    headers={"x-api-key": self.secret},
                    json={"target": nomor, "confidence": round(confidence, 2),
                          "secret_key": self.secret},
                    timeout=12,
                )
                ok = r.status_code == 200
            if ok:
                return True, "WA terkirim"
            print(f"[WA] gagal HTTP {r.status_code}: {r.text[:200]}")   # hanya di log server
            with self.lock:
                self.last_sent = prev if not is_test else self.last_sent
            return False, f"Gagal kirim WA (HTTP {r.status_code})"
        except Exception as e:
            print(f"[WA] error: {e}")
            with self.lock:
                self.last_sent = prev if not is_test else self.last_sent
            return False, "Gagal menghubungi server WA"


# =====================================================================
# 6. PEMROSES AUDIO REAL-TIME (JALAN TERUS DI BACKGROUND)
# =====================================================================
class BabyCryProcessor(AudioProcessorBase):
    def __init__(self):
        self.level = 0.0
        self.progress = 0.0
        self.last_label = "Menunggu audio…"
        self.results = queue.Queue()
        self.nomor = ""
        self.wa = None
        self._buf, self._buf_len, self._sr = [], 0, None
        self._jobs = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def configure(self, nomor, wa):
        self.nomor, self.wa = nomor, wa

    def on_ended(self):
        self._stop.set()

    def recv_queued(self, frames):
        for frame in frames:
            self._ingest(frame)
        return frames

    def _ingest(self, frame):
        arr = frame.to_ndarray()
        ch = max(1, len(frame.layout.channels))
        mono = arr.mean(axis=0) if frame.format.is_planar else arr.reshape(-1, ch).mean(axis=1)
        if np.issubdtype(arr.dtype, np.integer):
            mono = mono / 32768.0
        mono = mono.astype(np.float32)

        sr = frame.sample_rate
        if self._sr != sr:
            self._buf, self._buf_len = [], 0
        self._sr = sr

        if mono.size:
            self.level = 0.7 * self.level + 0.3 * float(np.sqrt(np.mean(mono ** 2)))
        self._buf.append(mono)
        self._buf_len += mono.size
        target = sr * DURATION
        self.progress = min(1.0, self._buf_len / target)

        if self._buf_len >= target:
            data = np.concatenate(self._buf)
            chunk, rest = data[:target], data[target:]
            self._buf = [rest] if rest.size else []
            self._buf_len = rest.size
            try:
                self._jobs.put_nowait((chunk, sr))
            except queue.Full:
                pass  # analisis masih sibuk, lewati chunk ini

    def _worker(self):
        while not self._stop.is_set():
            try:
                chunk, sr = self._jobs.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if float(np.sqrt(np.mean(chunk ** 2))) < SILENCE_RMS:
                    self.last_label = f"{jam()} · senyap"
                    continue
                y = chunk if sr == SAMPLE_RATE else librosa.resample(
                    chunk, orig_sr=sr, target_sr=SAMPLE_RATE)
                pred, p_cry, p_not = classify(y)

                wa_ok, wa_info = False, "Tidak dikirim (bukan tangisan)"
                if pred == 1:
                    if self.wa and self.nomor:
                        wa_ok, wa_info = self.wa.send(self.nomor, p_cry * 100)
                    else:
                        wa_info = "Nomor WA belum diatur"

                self.last_label = (
                    f"{jam()} · {'TANGISAN' if pred else 'aman'} "
                    f"({(p_cry if pred else p_not) * 100:.0f}%)"
                )
                self.results.put({
                    "waktu": jam(), "sumber": "Mikrofon", "pred": pred,
                    "p_nangis": p_cry * 100, "p_bukan": p_not * 100,
                    "wa_ok": wa_ok, "wa_info": wa_info,
                })
            except Exception as e:
                print(f"[Worker] error: {e}")
                self.last_label = f"{jam()} · gagal menganalisis"


# =====================================================================
# 7. STATE & HELPER TAMPILAN
# =====================================================================
for key, default in {
    "riwayat": [], "notifs": [], "wa": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.wa is None:
    st.session_state.wa = WAClient()
wa_client: WAClient = st.session_state.wa


def push_notif(title: str, body: str, kind: str = "ok"):
    st.session_state.notifs.append(
        {"ts": time.time(), "jam": jam(), "title": title, "body": body, "kind": kind})


def push_event(item: dict):
    st.session_state.riwayat.insert(0, item)
    del st.session_state.riwayat[100:]
    if item["pred"] == 1:
        status = "WA terkirim ✓" if item["wa_ok"] else item["wa_info"]
        push_notif("🚨 Bayi terdeteksi menangis!",
                   f"Keyakinan {item['p_nangis']:.1f}% · {status}", "cry")


def render_notifs() -> str:
    now = time.time()
    st.session_state.notifs = [n for n in st.session_state.notifs if now - n["ts"] < NOTIF_TTL + 1]
    live = st.session_state.notifs[-3:]
    if not live:
        return ""
    cards = []
    for n in live:
        age = now - n["ts"]
        ico = "🚨" if n["kind"] == "cry" else "✅"
        cards.append(
            f'<div class="notif {n["kind"]}" style="animation-delay:-{age:.1f}s">'
            f'<div class="n-ico">{ico}</div><div style="flex:1;min-width:0">'
            f'<div class="n-head"><b>Baby Monitor</b><span>{n["jam"]}</span></div>'
            f'<div class="n-title">{html.escape(n["title"])}</div>'
            f'<div class="n-body">{html.escape(n["body"])}</div></div></div>')
    return f'<div class="notif-stack">{"".join(cards)}</div>'


def render_chat(items) -> str:
    if not items:
        return ('<div class="chat"><div class="empty">Belum ada deteksi.<br>'
                'Hidupkan mic, hasil akan muncul di sini seperti chat 💬</div></div>')
    parts = []
    for it in items[:30]:
        if it["pred"] == 1:
            tick = '<span class="tick ok">✓✓</span>' if it["wa_ok"] else '<span class="tick">✓</span>'
            parts.append(
                f'<div class="bubble cry"><div class="b-who">🚨 Tangisan Terdeteksi</div>'
                f'<div class="b-txt">Keyakinan model <b>{it["p_nangis"]:.1f}%</b><br>'
                f'<span style="color:#8696a0;font-size:12.5px">{html.escape(it["wa_info"])}</span></div>'
                f'<div class="b-meta">{it["waktu"]} · {html.escape(it["sumber"])}{tick}</div></div>')
        else:
            parts.append(
                f'<div class="pill">✅ {it["waktu"]} · Aman ({it["p_bukan"]:.0f}%) · '
                f'{html.escape(it["sumber"])}</div>')
    parts.append('<div class="pill">Log Deteksi</div>')
    return f'<div class="chat">{"".join(parts)}</div>'


# =====================================================================
# 8. PANEL LIVE (DIPERBARUI OTOMATIS TIAP 1 DETIK)
# =====================================================================
@st.fragment(run_every=1.0)
def live_panel(ctx, nomor: str, wa: WAClient):
    playing = bool(ctx is not None and ctx.state.playing)
    proc = ctx.audio_processor if playing else None

    if proc is not None:
        proc.configure(nomor, wa)
        while True:
            try:
                push_event(proc.results.get_nowait())
            except queue.Empty:
                break

    notif_html = render_notifs()
    if notif_html:
        st.markdown(notif_html, unsafe_allow_html=True)

    recent_cry = any(n["kind"] == "cry" for n in st.session_state.notifs)
    if not playing:
        cls, title, sub = "off", "Mikrofon nonaktif", "Klik “Hidupkan Mic” di atas untuk mulai memantau."
    elif recent_cry:
        cls, title, sub = "alert", "Bayi menangis!", proc.last_label if proc else ""
    else:
        cls, title, sub = "", "Mendengarkan…", proc.last_label if proc else ""
        if proc is not None:
            sub += f" · buffer {proc.progress * DURATION:.1f}/{DURATION}s"

    level = min(100, int((proc.level if proc else 0) * 500))
    total = len(st.session_state.riwayat)
    nangis = sum(1 for r in st.session_state.riwayat if r["pred"] == 1)
    terkirim = sum(1 for r in st.session_state.riwayat if r["wa_ok"])

    st.markdown(
        f'<div class="status {cls}"><div class="pulse"></div><div>'
        f'<div class="st-title">{title}</div><div class="st-sub">{html.escape(sub)}</div></div></div>'
        f'<div class="meter"><div style="width:{level}%"></div></div>'
        f'<div class="stats">'
        f'<div class="stat"><b>{total}</b><span>Analisis</span></div>'
        f'<div class="stat"><b>{nangis}</b><span>Tangisan</span></div>'
        f'<div class="stat"><b>{terkirim}</b><span>WA terkirim</span></div></div>'
        f'<div class="sec-title">💬 Log Deteksi</div>' + render_chat(st.session_state.riwayat),
        unsafe_allow_html=True,
    )


# =====================================================================
# 9. ANTARMUKA UTAMA
# =====================================================================
st.markdown(
    '<div class="appbar"><div class="avatar">👶</div><div>'
    '<div class="ab-title">Baby Monitor</div>'
    '<div class="ab-sub">Deteksi tangisan real-time · SVM RBF · Notifikasi WhatsApp</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

nomor_raw = st.text_input(
    "📱 Nomor WhatsApp tujuan notifikasi",
    placeholder="081234567890 atau 6281234567890",
)
nomor = "".join(filter(str.isdigit, nomor_raw))
if nomor.startswith("0"):
    nomor = "62" + nomor[1:]
nomor_valid = len(nomor) >= 10

if not wa_client.configured:
    st.warning("⚠️ Gateway WhatsApp belum dikonfigurasi di server (Secrets). Deteksi tetap jalan, notifikasi WA tidak terkirim.")

if st.button("📲 Test Kirim WA", disabled=not nomor_valid, use_container_width=True):
    with st.spinner("Mengirim pesan tes…"):
        ok, msg = wa_client.send(nomor, 99.99, is_test=True)
    push_notif("Test koneksi WA", msg, "ok" if ok else "cry")
    st.toast(msg, icon="✅" if ok else "❌")

# --- Kontrol mikrofon: klik SEKALI, hidup terus sampai dimatikan ---
ctx = None
if nomor_valid:
    ctx = webrtc_streamer(
        key="baby-mic",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=BabyCryProcessor,
        media_stream_constraints={
            "video": False,
            "audio": {                      # matikan filter browser agar suara speaker/jauh tidak dibuang
                "echoCancellation": False,
                "noiseSuppression": False,
                "autoGainControl": False,
                "channelCount": 1,
            },
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        translations={"start": "🎙️ Hidupkan Mic", "stop": "⏹️ Matikan Mic"},
        async_processing=True,
    )
else:
    st.info("🔒 Masukkan nomor WhatsApp yang valid untuk membuka tombol mikrofon.")

live_panel(ctx, nomor, wa_client)

# --- Uji file audio (alternatif untuk testing) ---
with st.expander("📂 Uji dengan file audio (.wav / .mp3)"):
    up = st.file_uploader("Unggah rekaman untuk pengujian:", type=["wav", "mp3"])
    if up is not None:
        st.audio(up)
        if st.button("🔍 Analisis file", disabled=not nomor_valid, use_container_width=True):
            with st.spinner("Mengekstrak fitur & klasifikasi…"):
                try:
                    y_up, _ = librosa.load(up, sr=SAMPLE_RATE, mono=True)
                    pred, p_cry, p_not = classify(y_up)
                    wa_ok, wa_info = False, "Tidak dikirim (bukan tangisan)"
                    if pred == 1:
                        wa_ok, wa_info = wa_client.send(nomor, p_cry * 100)
                    push_event({
                        "waktu": jam(), "sumber": f"File ({up.name})", "pred": pred,
                        "p_nangis": p_cry * 100, "p_bukan": p_not * 100,
                        "wa_ok": wa_ok, "wa_info": wa_info,
                    })
                    if pred == 0:
                        push_notif("Kondisi aman", f"Bukan tangisan ({p_not * 100:.1f}%)", "ok")
                except Exception as err:
                    st.error(f"Gagal memproses file audio: {err}")