import base64
import html
import io
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Abaikan warning format nama fitur scikit-learn agar log container tetap bersih
warnings.filterwarnings("ignore")

import joblib
import librosa
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components

# =====================================================================
# 1. KONFIGURASI
# =====================================================================
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_CHUNK = SAMPLE_RATE * DURATION
COOLDOWN_WA_DETIK = 30
NOTIF_TTL = 8
MAX_CHUNKS = 6          # maksimal potongan 5 detik yang dianalisis per rekaman
CHUNK_SECONDS = 10      # mic dianalisis setiap 10 detik
SILENCE_RMS = 0.002     # di bawah ini dianggap senyap (tidak dianalisis)
TZ = ZoneInfo("Asia/Jakarta")

st.set_page_config(
    page_title="Baby Monitor - Deteksi Tangisan",
    page_icon="👶",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def jam() -> str:
    return datetime.now(TZ).strftime("%H:%M:%S")


mic_stream = components.declare_component(
    "mic_stream", path=str(Path(__file__).parent / "mic_component")
)


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
.block-container{padding:.8rem .9rem 4rem !important; max-width:520px;}

.appbar{display:flex; align-items:center; gap:12px; padding:12px 14px; margin-bottom:12px;
  background:var(--panel); border-radius:16px; border:1px solid var(--line);}
.avatar{width:44px; height:44px; border-radius:50%; display:grid; place-items:center; font-size:24px;
  background:linear-gradient(135deg,#00a884,#0b6e5a);}
.ab-title{font-weight:700; font-size:17px; line-height:1.2;}
.ab-sub{font-size:12.5px; color:var(--muted);}

.stTextInput input{background:var(--bubble) !important; color:var(--text) !important;
  border:1px solid var(--line) !important; border-radius:12px !important;}
.stTextInput label p{color:var(--muted) !important; font-size:13px;}
.stButton>button{border-radius:12px; font-weight:600; border:1px solid var(--line);
  background:var(--bubble); color:var(--text); min-height:46px;}
.stButton>button:hover{border-color:var(--green); color:var(--green);}
[data-testid="stExpander"]{background:var(--panel); border:1px solid var(--line); border-radius:14px;}
[data-testid="stAudioInput"]{border-radius:14px;}

.status{display:flex; align-items:center; gap:14px; padding:16px; margin:10px 0 10px;
  background:var(--panel); border:1px solid var(--line); border-radius:16px;}
.pulse{width:14px; height:14px; border-radius:50%; flex:none; color:var(--green);
  background:currentColor; animation:pulse 1.6s infinite;}
.status.off .pulse{color:#667781; animation:none;}
.status.alert{border-color:var(--red); background:#2a1317;}
.status.alert .pulse{color:var(--red);}
.st-title{font-weight:700; font-size:15px;}
.st-sub{font-size:12.5px; color:var(--muted); margin-top:2px;}
@keyframes pulse{0%{box-shadow:0 0 0 0 currentColor;} 70%{box-shadow:0 0 0 14px transparent;} 100%{box-shadow:0 0 0 0 transparent;}}

.stats{display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:12px;}
.stat{background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:10px; text-align:center;}
.stat b{display:block; font-size:20px;}
.stat span{font-size:11.5px; color:var(--muted);}

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
  100%{transform:translateY(-130%); opacity:0; visibility:hidden;}
}

.chat{display:flex; flex-direction:column-reverse; gap:8px; max-height:460px; overflow-y:auto;
  padding:14px 10px; border-radius:16px; border:1px solid var(--line);
  background:
    radial-gradient(circle at 20% 20%, rgba(0,168,132,.06) 0, transparent 40%),
    radial-gradient(circle at 80% 70%, rgba(83,189,235,.05) 0, transparent 40%),
    #0d1a21;}
.bubble{max-width:88%; padding:9px 12px 6px; border-radius:4px 14px 14px 14px; background:var(--bubble);
  align-self:flex-start; box-shadow:0 1px 2px rgba(0,0,0,.4);}
.bubble.cry{border-left:4px solid var(--red);}
.b-who{font-size:12px; font-weight:700; color:#fca5a5;}
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


def classify(y: np.ndarray):
    """Rekaman panjang dipotong per 5 detik; ambil potongan dengan peluang tangisan tertinggi."""
    chunks = [y[i:i + SAMPLES_PER_CHUNK] for i in range(0, max(len(y), 1), SAMPLES_PER_CHUNK)]
    chunks = [c for c in chunks if len(c) >= SAMPLE_RATE] or [y]
    best = (0, 0.0, 1.0)
    for c in chunks[:MAX_CHUNKS]:
        fitur = scaler_audio.transform(extract_182_features(c, SAMPLE_RATE))
        pred = int(model_svm.predict(fitur)[0])
        proba = model_svm.predict_proba(fitur)[0]
        if float(proba[1]) >= best[1]:
            best = (pred, float(proba[1]), float(proba[0]))
    return best


# =====================================================================
# 5. NOTIFIKASI WHATSAPP (KONFIGURASI HANYA DARI st.secrets)
# =====================================================================
def _secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


WA_METHOD = _secret("WA_METHOD", "webhook").lower()
WEBHOOK_URL = _secret("WEBHOOK_URL")
WEBHOOK_SECRET = _secret("WEBHOOK_SECRET")
FONNTE_TOKEN = _secret("FONNTE_TOKEN")
WA_CONFIGURED = bool(FONNTE_TOKEN) if WA_METHOD == "fonnte" else bool(WEBHOOK_URL and WEBHOOK_SECRET)


def kirim_wa(nomor: str, confidence: float, is_test: bool = False):
    if not WA_CONFIGURED:
        return False, "Gateway WA belum dikonfigurasi di server."

    sisa = COOLDOWN_WA_DETIK - (time.time() - st.session_state.last_wa)
    if not is_test and sisa > 0:
        return False, f"Ditahan anti-spam ({int(sisa) + 1} dtk lagi)"

    pesan = (
        "🚨 *PERINGATAN DETEKSI TANGISAN BAYI* 🚨\n\n"
        "Sistem mendeteksi suara bayi sedang menangis!\n"
        f"• *Status:* {'UJI COBA KONEKSI (TEST)' if is_test else 'Bayi Menangis'}\n"
        "• *Model:* SVM (RBF Kernel)\n"
        f"• *Keyakinan:* {confidence:.2f}%\n"
        f"• *Waktu:* {jam()} WIB\n\n"
        "Mohon segera periksa kondisi bayi Anda."
    )
    try:
        if WA_METHOD == "fonnte":
            r = requests.post(
                "https://api.fonnte.com/send",
                headers={"Authorization": FONNTE_TOKEN},
                data={"target": nomor, "message": pesan},
                timeout=12,
            )
            ok = r.status_code == 200 and bool(r.json().get("status"))
        else:
            r = requests.post(
                WEBHOOK_URL,
                headers={"x-api-key": WEBHOOK_SECRET},
                json={"target": nomor, "message": pesan},  # <-- Ubah baris ini jadi "message": pesan
                timeout=12,
            )
            ok = r.status_code == 200
        if ok:
            if not is_test:
                st.session_state.last_wa = time.time()
            return True, "WA terkirim"
        print(f"[WA] gagal HTTP {r.status_code}: {r.text[:200]}")  # hanya di log server
        return False, f"Gagal kirim WA (HTTP {r.status_code})"
    except Exception as e:
        print(f"[WA] error: {e}")
        return False, "Gagal menghubungi server WA"


# =====================================================================
# 6. STATE & HELPER TAMPILAN
# =====================================================================
for k, v in {"riwayat": [], "notifs": [], "last_wa": 0.0, "mic_active": False,
             "last_chunk_id": None, "last_info": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
    else:
        push_notif("Kondisi aman", f"Bukan tangisan ({item['p_bukan']:.1f}%)", "ok")


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
                'Rekam suara, hasil akan muncul di sini seperti chat 💬</div></div>')
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


def analisis(y: np.ndarray, sumber: str, nomor: str):
    pred, p_cry, p_not = classify(y)
    wa_ok, wa_info = False, "Tidak dikirim (bukan tangisan)"
    if pred == 1:
        wa_ok, wa_info = kirim_wa(nomor, p_cry * 100)
    push_event({
        "waktu": jam(), "sumber": sumber, "pred": pred,
        "p_nangis": p_cry * 100, "p_bukan": p_not * 100,
        "wa_ok": wa_ok, "wa_info": wa_info,
    })


# =====================================================================
# 7. ANTARMUKA UTAMA
# =====================================================================
st.markdown(
    '<div class="appbar"><div class="avatar">👶</div><div>'
    '<div class="ab-title">Baby Monitor</div>'
    '<div class="ab-sub">Deteksi tangisan · SVM RBF · Notifikasi WhatsApp</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

nomor_raw = st.text_input(
    "📱 Nomor WhatsApp tujuan notifikasi",
    placeholder="081234567890 atau 6281234567890",
    disabled=st.session_state.mic_active,
)
nomor = "".join(filter(str.isdigit, nomor_raw))
if nomor.startswith("0"):
    nomor = "62" + nomor[1:]
nomor_valid = len(nomor) >= 10

if not WA_CONFIGURED:
    st.warning("⚠️ Gateway WhatsApp belum dikonfigurasi di server (Secrets). Deteksi tetap jalan, notifikasi WA tidak terkirim.")

if not nomor_valid:
    st.info("🔒 Masukkan nomor WhatsApp yang valid untuk membuka tombol mikrofon.")

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🎙️ Hidupkan Mic", type="primary", use_container_width=True,
                 disabled=(not nomor_valid or st.session_state.mic_active)):
        st.session_state.mic_active = True
        st.session_state.last_chunk_id = None
        st.session_state.last_info = "Menunggu potongan audio pertama…"
        st.rerun()
with c2:
    if st.button("⏹️ Matikan Mic", use_container_width=True,
                 disabled=(not st.session_state.mic_active)):
        st.session_state.mic_active = False
        st.session_state.last_chunk_id = None
        st.rerun()

if "test_wa_cooldown_until" not in st.session_state:
    st.session_state.test_wa_cooldown_until = 0.0

with c3:
    sisa_awal = max(0, int(st.session_state.test_wa_cooldown_until - time.time()))
    
    @st.fragment(run_every=1 if sisa_awal > 0 else None)
    def render_tombol_test_wa():
        sisa_detik = max(0, int(st.session_state.test_wa_cooldown_until - time.time()))
        sedang_cooldown = sisa_detik > 0

        label_tombol = f"⏳ {sisa_detik}s" if sedang_cooldown else "📲 Test WA"

        if st.button(
            label_tombol,
            disabled=(not nomor_valid or sedang_cooldown),
            use_container_width=True
        ):
            with st.spinner("Mengirim pesan tes…"):
                ok, msg = kirim_wa(nomor, 99.99, is_test=True)
            push_notif("Test koneksi WA", msg, "ok" if ok else "cry")
            
            st.session_state.test_wa_cooldown_until = time.time() + 30
            st.rerun(scope="app")  # <--- Ubah jadi scope="app" agar timer & notif luar langsung aktif

        if sisa_awal > 0 and sisa_detik == 0:
            st.rerun(scope="app")  # <--- Ubah jadi scope="app" untuk mematikan timer saat habis

    render_tombol_test_wa()

# Mic aktif: komponen merekam terus di browser, mengirim potongan tiap CHUNK_SECONDS detik
if st.session_state.mic_active and nomor_valid:
    data = mic_stream(chunk_seconds=CHUNK_SECONDS, key="mic_stream", default=None)
    if data and data.get("id") != st.session_state.last_chunk_id:
        st.session_state.last_chunk_id = data["id"]
        try:
            raw = base64.b64decode(data["wav"])
            y, _ = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True)
            if float(np.sqrt(np.mean(y ** 2))) < SILENCE_RMS:
                st.session_state.last_info = f"{jam()} · senyap, tidak dianalisis"
            else:
                analisis(y, "Mikrofon", nomor)
                r = st.session_state.riwayat[0]
                st.session_state.last_info = (
                    f"{r['waktu']} · {'TANGISAN' if r['pred'] else 'aman'}"
                )
        except Exception as err:
            print(f"[Mic] error: {err}")
            st.session_state.last_info = f"{jam()} · gagal menganalisis potongan audio"

# Notifikasi melayang + status + log
notif_html = render_notifs()
if notif_html:
    st.markdown(notif_html, unsafe_allow_html=True)

riwayat = st.session_state.riwayat
recent_cry = any(n["kind"] == "cry" for n in st.session_state.notifs)
mic_on = st.session_state.mic_active
if recent_cry:
    cls, title, sub = "alert", "Bayi menangis!", st.session_state.last_info or f"Terakhir: {riwayat[0]['waktu']}"
elif mic_on:
    cls, title, sub = "", "Mendengarkan…", f"{st.session_state.last_info} · analisis tiap {CHUNK_SECONDS} dtk"
elif not riwayat:
    cls, title, sub = "off", "Mikrofon nonaktif", "Klik “Hidupkan Mic” untuk mulai memantau."
else:
    cls, title, sub = "off", "Mikrofon nonaktif", f"Terakhir dicek: {riwayat[0]['waktu']}"

total = len(riwayat)
nangis = sum(1 for r in riwayat if r["pred"] == 1)
terkirim = sum(1 for r in riwayat if r["wa_ok"])

st.markdown(
    f'<div class="status {cls}"><div class="pulse"></div><div>'
    f'<div class="st-title">{title}</div><div class="st-sub">{html.escape(sub)}</div></div></div>'
    f'<div class="stats">'
    f'<div class="stat"><b>{total}</b><span>Analisis</span></div>'
    f'<div class="stat"><b>{nangis}</b><span>Tangisan</span></div>'
    f'<div class="stat"><b>{terkirim}</b><span>WA terkirim</span></div></div>'
    f'<div class="sec-title">💬 Log Deteksi</div>' + render_chat(riwayat),
    unsafe_allow_html=True,
)

# Uji file audio
with st.expander("📂 Uji dengan file audio (.wav / .mp3)"):
    up = st.file_uploader("Unggah rekaman untuk pengujian:", type=["wav", "mp3"])
    if up is not None:
        st.audio(up)
        if st.button("🔍 Analisis file", disabled=not nomor_valid, use_container_width=True):
            with st.spinner("Menganalisis file…"):
                try:
                    y_up, _ = librosa.load(up, sr=SAMPLE_RATE, mono=True)
                    analisis(y_up, f"File ({up.name})", nomor)
                    st.rerun()
                except Exception as err:
                    st.error(f"Gagal memproses file audio: {err}")