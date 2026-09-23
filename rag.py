"""RAG (jurnal skincare) + rekomendasi produk dari Supabase."""
import re
from operator import itemgetter
 
import nltk
import pandas as pd
import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import NLTKTextSplitter
from supabase import create_client
 
# ============================================================
# KONFIGURASI
# ============================================================
CHAT_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"
 
PDF_BUCKET = "Journal"
PDF_FILE = "jurnal skincare.pdf"
PRODUCT_TABLE = "skincare_cleaned"
INGREDIENT_COL = None  # isi manual jika deteksi otomatis salah, mis. "ingredients"
 
# Label model CNN -> istilah di jurnal (untuk query retriever)
JOURNAL_TERMS = {
    "acne": "acne",
    "blackheads": "acne (comedonal / blackheads) and oily skin",
    "dark_spots": "dark spots",
    "pores": "large pores",
    "wrinkles": "fine lines and wrinkles",
}
 
# Bahan yang mencapai konsensus per kondisi + % dermatolog yang merekomendasikan
# (Table II jurnal Alvarez et al., JAAD 2025)
CONSENSUS_INGREDIENTS = {
    "acne": {"retinoids": 96.8, "benzoyl peroxide": 95.2, "salicylic acid": 93.6,
             "clindamycin": 90.3, "azelaic acid": 87.1, "glycolic acid": 79.0},
    # jurnal tidak membahas blackheads -> memakai kategori acne (& oily skin)
    "blackheads": {"retinoids": 96.8, "benzoyl peroxide": 95.2, "salicylic acid": 93.6,
                   "azelaic acid": 87.1, "glycolic acid": 79.0},
    "dark_spots": {"hydroquinone": 98.4, "retinoids": 96.8, "kojic acid": 93.6,
                   "glycolic acid": 91.9, "azelaic acid": 88.7, "vitamin c": 87.1,
                   "tranexamic acid": 87.1, "niacinamide": 79.0},
    "pores": {"retinoids": 93.6},
    "wrinkles": {"mineral sunscreen": 96.8, "retinoids": 96.8, "vitamin c": 88.7,
                 "chemical sunscreen": 82.3},
}
 
# Bahan yang umumnya hanya tersedia dengan resep dokter
PRESCRIPTION_ONLY = {"clindamycin", "hydroquinone"}
 
# Nama yang biasa muncul di daftar komposisi produk untuk tiap bahan
INGREDIENT_KEYWORDS = {
    "retinoids": ["retinol", "retinal", "retinaldehyde", "retinyl", "tretinoin",
                  "adapalene", "tazarotene", "retinoate", "retinoid"],
    "salicylic acid": ["salicylic acid", "betaine salicylate", "willow bark"],
    "benzoyl peroxide": ["benzoyl peroxide"],
    "azelaic acid": ["azelaic acid", "potassium azeloyl diglycinate"],
    "clindamycin": ["clindamycin"],
    "glycolic acid": ["glycolic acid"],
    "hydroquinone": ["hydroquinone"],
    "kojic acid": ["kojic acid", "kojic dipalmitate"],
    "vitamin c": ["ascorbic acid", "ascorbyl", "ascorbate", "vitamin c"],
    "tranexamic acid": ["tranexamic acid"],
    "niacinamide": ["niacinamide", "nicotinamide"],
    "mineral sunscreen": ["zinc oxide", "titanium dioxide"],
    "chemical sunscreen": ["avobenzone", "octinoxate", "octocrylene", "homosalate",
                           "oxybenzone", "octisalate", "ethylhexyl methoxycinnamate",
                           "ethylhexyl salicylate", "butyl methoxydibenzoylmethane",
                           "bis-ethylhexyloxyphenol methoxyphenyl triazine",
                           "diethylamino hydroxybenzoyl hexyl benzoate",
                           "ethylhexyl triazone", "tinosorb", "uvinul"],
}
def normalize_text(value):
    """Samakan tanda '-' dan '_' dengan spasi, huruf kecil, spasi ganda dirapikan.
    Dipakai untuk pencocokan bahan maupun untuk tampilan."""
    return re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", str(value).lower())).strip()
 
 
# pola dibuat dari keyword yang sudah dinormalisasi, supaya "alpha-arbutin"
# dan "alpha arbutin" sama-sama terdeteksi
INGREDIENT_PATTERNS = {
    ing: re.compile("|".join(r"\b" + re.escape(normalize_text(k)) + r"\b" for k in kws))
    for ing, kws in INGREDIENT_KEYWORDS.items()
}
 
# Kolom yang dicari otomatis di tabel Supabase (urutan = prioritas)
COLUMN_HINTS = {
    "image": ["image", "img", "picture", "photo", "thumbnail"],
    "link": ["url", "link", "href"],
    "name": ["product_name", "name", "title", "produk"],
    "brand": ["brand", "merek", "merk"],
    "price": ["price", "harga"],
}
 
 
SYSTEM_PROMPT = """Kamu adalah asisten edukasi skincare.
Jawab HANYA berdasarkan konteks dari jurnal "Skincare ingredients recommended by
cosmetic dermatologists: A Delphi consensus study" (Alvarez et al., JAAD 2025).
Jangan menambahkan informasi dari luar konteks. Jika informasi tidak ada di konteks,
katakan terus terang.
 
Untuk setiap kondisi kulit yang terdeteksi:
1. Sebutkan bahan (ingredient) yang mencapai konsensus beserta persentase dermatolog
   yang merekomendasikannya, jika tersedia di konteks.
2. Jelaskan singkat bukti ilmiahnya menurut jurnal.
Jurnal tidak membahas blackheads secara khusus; jika kondisinya blackheads, jelaskan
bahwa rekomendasi diambil dari kategori acne dan oily skin.
Jawab dalam Bahasa Indonesia, ringkas, dan akhiri dengan pengingat bahwa ini bukan
diagnosis medis serta sarankan konsultasi ke dokter kulit.

3. Tampilkan 5 rekomendasi produk, preview link rekomendasi produk, dan gambar produk, jangan melakukan generate gambar, ambil gambar pada link rekomendasi produk.

 """
 
 
# ============================================================
# SUPABASE
# ============================================================
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"].rstrip("/")
    return create_client(url, st.secrets["SUPABASE_KEY"])
 
 
@st.cache_data(ttl=600, show_spinner="Mengambil data produk...")
def load_products():
    rows, start, step = [], 0, 1000
    while True:  # ambil per 1000 baris sampai habis
        batch = (get_supabase().table(PRODUCT_TABLE).select("*")
                 .range(start, start + step - 1).execute().data)
        rows.extend(batch)
        if len(batch) < step:
            break
        start += step
    return pd.DataFrame(rows)
 
 
def find_ingredient_col(df):
    if INGREDIENT_COL:
        return INGREDIENT_COL
    for col in df.columns:
        if any(k in col.lower() for k in ["ingredient", "komposisi", "bahan", "composition"]):
            return col
    raise ValueError(
        f"Kolom ingredient tidak ditemukan. Kolom yang ada: {list(df.columns)}. "
        "Isi INGREDIENT_COL di rag.py secara manual."
    )
 
 
def detect_columns(df):
    """Tebak kolom gambar, link, nama, brand, dan harga dari nama kolom tabel."""
    found, used = {}, set()
    for key, hints in COLUMN_HINTS.items():
        col = next((c for c in df.columns
                    if c not in used and any(h in c.lower() for h in hints)), None)
        found[key] = col
        if col:
            used.add(col)
    return found
 
 
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_preview_image(url):
    """Ambil gambar utama (og:image) dari halaman produk. None jika gagal."""
    try:
        html = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}).text
        m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', html)
        return m.group(1) if m else None
    except Exception:
        return None
 
 
# ============================================================
# LOGIKA BERSAMA
# ============================================================
def get_detected(scores, threshold=0.5):
    """Kelas dengan skor >= threshold; minimal kelas dengan skor tertinggi."""
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [c for c, s in ranked if s >= threshold] or [ranked[0][0]]
 
 
def get_target_ingredients(scores):
    """Daftar (bahan, persen) dari jurnal untuk kondisi terdeteksi, urut dari konsensus tertinggi."""
    merged = {}
    for cond in get_detected(scores):
        for ing, pct in CONSENSUS_INGREDIENTS[cond].items():
            merged[ing] = max(pct, merged.get(ing, 0))
    return sorted(merged.items(), key=lambda x: -x[1])
 
 
# ============================================================
# REKOMENDASI PRODUK
# ============================================================
def recommend_products(scores, top_n=6):
    """Untuk tiap bahan dari jurnal, cari produk di Supabase yang mengandungnya.
 
    Return: list of dict {"ingredient", "pct", "prescription", "products" (DataFrame), "cols"}
    """
    targets = get_target_ingredients(scores)
    target_names = [ing for ing, _ in targets]
    df = load_products()
 
    cols = detect_columns(df) if not df.empty else {}
    ing_col = find_ingredient_col(df) if not df.empty else None
 
    if ing_col:
        # '-' disamakan dengan ' ' sebelum dicocokkan
        text = df[ing_col].fillna("").map(normalize_text)
        matched = text.apply(lambda t: [i for i in target_names if INGREDIENT_PATTERNS[i].search(t)])
 
        base = df.drop(columns=[ing_col]).copy()
        # rapikan kolom teks untuk tampilan (kecuali link & gambar)
        keep_raw = {cols.get("link"), cols.get("image")}
        for col in base.columns:
            # dtype "object" (pandas 2) atau "str" (pandas 3)
            if col not in keep_raw and pd.api.types.is_string_dtype(base[col]):
                base[col] = base[col].fillna("").map(normalize_text).str.title()
        base.insert(0, "bahan_sesuai_jurnal", matched.apply(", ".join))
        base["_jumlah"] = matched.apply(len)
 
    results = []
    for ing, pct in targets:
        if ing_col:
            mask = matched.apply(lambda m: ing in m)
            products = (base[mask]
                        .sort_values("_jumlah", ascending=False)  # produk multi-bahan di atas
                        .drop(columns=["_jumlah"])
                        .head(top_n)
                        .reset_index(drop=True))
        else:
            products = pd.DataFrame()
        results.append({
            "ingredient": ing,
            "pct": pct,
            "prescription": ing in PRESCRIPTION_ONLY,
            "products": products,
            "cols": cols,
        })
    return results
 
 
# ============================================================
# RAG
# ============================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
 
 
@st.cache_resource(show_spinner="Menyiapkan basis pengetahuan dari jurnal...")
def get_retriever():
    nltk.download("punkt_tab", quiet=True)
    pdf_url = get_supabase().storage.from_(PDF_BUCKET).get_public_url(PDF_FILE)
    pages = PyPDFLoader(pdf_url).load_and_split()
    splitter = NLTKTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
 
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_HF,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return db.as_retriever(search_kwargs={"k": 10})
 
 
def get_rag_chain(api_key):  # tidak di-cache supaya API key pengguna tidak tersimpan
    chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model=CHAT_MODEL)
    template = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(
            "Konteks dari jurnal:\n{context}\n\n"
            "Hasil deteksi model:\n{skin_condition}\n\n"
            "Jawaban:"
        ),
    ])
    return (
        {
            "context": itemgetter("query") | get_retriever() | format_docs,
            "skin_condition": itemgetter("skin_condition"),
        }
        | template
        | chat_model
        | StrOutputParser()
    )
 
 
def build_inputs(scores):
    detected = get_detected(scores)
    terms = [JOURNAL_TERMS[c] for c in detected]
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    table = "\n".join(f"- {c}: {s * 100:.2f}%" for c, s in ranked)
    return {
        "query": "Recommended skincare ingredients for " + ", ".join(terms),
        "skin_condition": f"Kondisi utama: {', '.join(terms)}\nSkor model per kelas:\n{table}",
    }
 
 
def stream_answer(scores, api_key):
    return get_rag_chain(api_key).stream(build_inputs(scores))