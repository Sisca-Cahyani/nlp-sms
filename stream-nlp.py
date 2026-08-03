import re
import pickle

import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Cek SMS Penipuan",
    page_icon="🛡️",
    layout="centered",
)

# CSS - font kustom, skema warna, dan tampilan yang responsif untuk mobile
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .main .block-container {
            max-width: 680px;
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }

        /* Header */
        .app-badge {
            display: inline-block;
            background: #eef2ff;
            color: #4338ca;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.9rem;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
            color: #111827;
            line-height: 1.2;
        }
        .app-subtitle {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 2rem;
            font-weight: 400;
        }

        /* Input label */
        .stTextArea label p {
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            color: #374151 !important;
        }
        .stTextArea textarea {
            font-size: 0.98rem;
            border-radius: 12px !important;
            border: 1.5px solid #e5e7eb !important;
            padding: 0.9rem !important;
        }
        .stTextArea textarea:focus {
            border-color: #4f46e5 !important;
            box-shadow: 0 0 0 1px #4f46e5 !important;
        }

        /* Button */
        .stButton button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 0.98rem !important;
            padding: 0.6rem 1.4rem !important;
            background-color: #4338ca !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(67, 56, 202, 0.25) !important;
            transition: all 0.15s ease-in-out !important;
        }
        .stButton button:hover {
            background-color: #3730a3 !important;
            box-shadow: 0 6px 16px rgba(67, 56, 202, 0.35) !important;
            transform: translateY(-1px);
        }

        /* Result card */
        .result-card {
            border-radius: 16px;
            padding: 1.5rem 1.6rem;
            margin-top: 1.4rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .result-icon {
            font-size: 2rem;
            line-height: 1;
        }
        .result-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.2rem;
            opacity: 0.75;
        }
        .result-value {
            font-size: 1.4rem;
            font-weight: 700;
        }

        .card-fraud {
            background: #fef2f2;
            border: 1.5px solid #fecaca;
            color: #991b1b;
        }
        .card-normal {
            background: #f0fdf4;
            border: 1.5px solid #bbf7d0;
            color: #166534;
        }
        .card-promo {
            background: #eff6ff;
            border: 1.5px solid #bfdbfe;
            color: #1e40af;
        }

        /* Footer caption */
        .app-footer {
            font-size: 0.82rem;
            color: #9ca3af;
            margin-top: 1.8rem;
            line-height: 1.5;
        }

        /* Mobile tweaks */
        @media (max-width: 640px) {
            .app-title { font-size: 1.6rem; }
            .app-subtitle { font-size: 0.92rem; }
            .result-value { font-size: 1.2rem; }
            .result-icon { font-size: 1.6rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# RESOURCE LOADING (di-cache agar tidak diproses ulang tiap interaksi)
# =========================================================
@st.cache_resource
def load_nltk_stopwords():
    nltk.download("stopwords", quiet=True)
    stopwords_ind = stopwords.words("indonesian")
    more_stopword = ["tsel", "gb", "rb", "btw"]
    return stopwords_ind + more_stopword


@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


@st.cache_resource
def load_key_norm():
    return pd.read_csv("key_norm.csv")


@st.cache_resource
def load_model_and_vocab():
    with open("model_fraud.sav", "rb") as f:
        model = pickle.load(f)
    with open("new_selected_feature_tf-idf.sav", "rb") as f:
        vocab = pickle.load(f)
    return model, vocab


stopwords_ind = load_nltk_stopwords()
stemmer = load_stemmer()
key_norm = load_key_norm()
model, selected_vocab = load_model_and_vocab()


# =========================================================
# TEXT PREPROCESSING (harus identik dengan proses training di notebook)
# =========================================================
def casefolding(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[-+]?[0-9]+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    return text


def text_normalize(text):
    text = " ".join(
        [
            key_norm[key_norm["singkat"] == word]["hasil"].values[0]
            if (key_norm["singkat"] == word).any()
            else word
            for word in text.split()
        ]
    )
    text = str.lower(text)
    return text


def remove_stop_word(text):
    clean_words = [word for word in text.split() if word not in stopwords_ind]
    return " ".join(clean_words)


def stemming(text):
    return stemmer.stem(text)


def text_preprocessing_process(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_word(text)
    text = stemming(text)
    return text


LABEL_MAP = {
    0: "SMS Normal",
    1: "SMS Penipuan",
    2: "SMS Promo",
}

RESULT_STYLE = {
    0: {"icon": "✅", "css_class": "card-normal"},
    1: {"icon": "⚠️", "css_class": "card-fraud"},
    2: {"icon": "🏷️", "css_class": "card-promo"},
}


def predict_sms(raw_text):
    clean_text = text_preprocessing_process(raw_text)

    loaded_vec = TfidfVectorizer(
        decode_error="replace",
        vocabulary=set(selected_vocab),
    )
    vectorized = loaded_vec.fit_transform([clean_text])
    prediction = model.predict(vectorized)[0]

    label = LABEL_MAP.get(prediction, "Tidak diketahui")
    style = RESULT_STYLE.get(prediction, {"icon": "❔", "css_class": "card-normal"})

    return label, style, clean_text


# =========================================================
# UI
# =========================================================
st.markdown('<div class="app-badge">Keamanan Digital</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Deteksi SMS Penipuan</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Sistem klasifikasi SMS berbasis kecerdasan buatan</div>',
    unsafe_allow_html=True,
)

sms_input = st.text_area(
    "Masukkan teks SMS",
    height=140,
    placeholder="Contoh: Selamat, nomor Anda terpilih sebagai pemenang undian...",
    label_visibility="visible",
)

submitted = st.button("Periksa SMS", type="primary", use_container_width=True)

if submitted:
    if not sms_input.strip():
        st.warning("Silakan masukkan teks SMS terlebih dahulu.")
    else:
        with st.spinner("Memeriksa SMS..."):
            hasil, style, clean_text = predict_sms(sms_input)

        st.markdown(
            f"""
            <div class="result-card {style['css_class']}">
                <div class="result-icon">{style['icon']}</div>
                <div>
                    <div class="result-label">Hasil Pemeriksaan</div>
                    <div class="result-value">{hasil}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Lihat detail teks setelah diproses"):
            st.write(clean_text if clean_text else "(kosong setelah preprocessing)")

st.markdown(
    '<div class="app-footer">Alat bantu untuk mengenali SMS mencurigakan secara otomatis. '
    "Hasil pemeriksaan bersifat estimasi dan bukan keputusan final.</div>",
    unsafe_allow_html=True,
)