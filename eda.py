import os

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st

from supabase import create_client

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["axes.titleweight"] = "bold"

def _get_secret(key: str) -> str | None:
    """Read a secret from st.secrets if available, else fall back to an
    environment variable. st.secrets raises (rather than just missing the
    key) when no secrets.toml exists at all, e.g. in local dev, so this
    guards against that instead of crashing the whole page."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:  # noqa: BLE001 - no secrets.toml configured at all
        pass
    return os.environ.get(key)


SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY")

LOCAL_FALLBACK_CSV = "data_produk.csv"
TABLE_NAME = "skincare_cleaned"

# Data loading

@st.cache_data(show_spinner="Memuat data katalog produk...")
def load_data() -> pd.DataFrame:

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            all_rows = []
            batch_size = 1000
            start = 0
            while True:
                response = (
                    supabase.table(TABLE_NAME)
                    .select("*")
                    .range(start, start + batch_size - 1)
                    .execute()
                )
                batch = response.data
                if not batch:
                    break
                all_rows.extend(batch)
                start += batch_size

            if all_rows:
                return pd.DataFrame(all_rows)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not fatal
            st.warning(
                f"Tidak dapat memuat data dari Supabase ({exc}). "
                "Menggunakan file lokal `data_produk.csv` sebagai cadangan."
            )

    return pd.read_csv(LOCAL_FALLBACK_CSV)


def _render_overview(df: pd.DataFrame) -> None:
    st.subheader("Gambaran Umum Data")
    st.write(
        f"Dataset katalog produk ini terdiri dari **{df.shape[0]:,} baris** "
        f"dan **{df.shape[1]} kolom**."
    )
    st.dataframe(df.head(10), width="stretch")


def _render_data_quality(df: pd.DataFrame) -> None:
    st.subheader("Kualitas Data")

    quality = pd.DataFrame(
        {
            "missing_values": df.isnull().sum(),
            "missing_pct": (df.isnull().mean() * 100).round(2),
            "unique_values": df.nunique(),
            "dtype": df.dtypes.astype(str),
        }
    )
    st.write(f"Baris yang sepenuhnya duplikat: **{df.duplicated().sum()}**")
    st.dataframe(quality, width="stretch")

    st.markdown(
        """
**Insight kualitas data:**
- Tidak ada *missing values* dan duplikat pada baris, menandakan dataset sudah bersih dan siap digunakan.
- Setiap tautan produk mengarah ke satu domain yang sama, artinya rekomendasi yang dihasilkan hanya akan merefleksikan katalog dan harga dari **satu retailer** tersebut.
- Data tidak memiliki kolom harga, rating, ataupun jumlah review, sehingga kita belum bisa merangking produk berdasarkan popularitas maupun harga — setiap produk yang cocok memiliki *"nilai yang setara"* saat direkomendasikan.
"""
    )


def _render_problem_distribution(df: pd.DataFrame) -> None:
    st.subheader("Berapa Banyak Produk yang Dimiliki Setiap Masalah Kulit?")

    problem_counts = df["problem"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(problem_counts.index, problem_counts.values, color=sns.color_palette("Set2"))
    ax.bar_label(bars, padding=3, fontweight="bold")
    ax.set_title("Number of Catalog Rows (Problem–Ingredient–Product Links) per Skin Problem")
    ax.set_xlabel("Skin Problem")
    ax.set_ylabel("Number of Rows")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
**Insight Bisnis:** Kelima masalah kulit yang akan didukung (*comedo, wrinkles, acne, dark spots,* dan *pores*) memiliki jumlah data yang cukup baik. Kategori dengan jumlah data paling sedikit, yaitu *pores*, masih tergolong cukup besar untuk digunakan. Tidak ada kategori masalah kulit yang datanya terlalu sedikit, sehingga mesin rekomendasi dapat dikembangkan untuk **kelima masalah kulit** tanpa perlu menambahkan data terlebih dahulu.

Kategori *comedo* dan *wrinkles* memiliki jumlah data paling banyak. Namun, hal ini bisa saja disebabkan oleh banyaknya bahan/kandungan yang dilacak oleh situs sumber untuk masalah kulit tersebut, bukan karena tingginya permintaan pasar secara nyata — perlu divalidasi kembali dengan tim produk.
"""
    )


def _render_top_ingredients(df: pd.DataFrame) -> None:
    st.subheader("Bahan Aktif Paling Sering Muncul di Katalog")

    top_ing = (
        df["ingredient"]
        .value_counts()
        .head(15)
        .sort_values(ascending=True)
        .rename_axis("ingredient")
        .reset_index(name="count")
    )

    fig = px.bar(
        top_ing,
        x="count",
        y="ingredient",
        orientation="h",
        text="count",
        title="Top 15 Most-Listed Ingredients in the Catalog",
        color_discrete_sequence=["#66C2A5"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=550,
        margin=dict(l=10, r=30, t=60, b=10),
        yaxis=dict(title="", automargin=True),
        xaxis=dict(title="Number of Product Rows"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
**Insight Bisnis:** Beberapa bahan memiliki jumlah data pada angka bulat (misalnya 100 atau 50), yang mengindikasikan katalog sumber kemungkinan dikumpulkan menggunakan **kuota tetap produk untuk setiap bahan**, bukan murni popularitas pasar. Untuk Soluskin, sistem pemeringkatan/sampling dipastikan kembali untuk melihat dari komposisi ingredient produk, sehingga pelanggan dengan masalah kulit yang sama tidak berisiko terus menerima rekomendasi produk yang serupa.
"""
    )


def _render_top_products(df: pd.DataFrame) -> None:
    st.subheader("Top 15 Produk Berdasarkan Jumlah Kemunculan")

    product_counts = (
        df["product_name"]
        .value_counts()
        .head(15)
        .sort_values(ascending=True)
        .rename_axis("product_name")
        .reset_index(name="count")
    )


    chart_height = 120 + 35 * len(product_counts)

    fig = px.bar(
        product_counts,
        x="count",
        y="product_name",
        orientation="h",
        text="count",
        title="Top 15 Products",
        color="count",
        color_continuous_scale="PuRd",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=chart_height,
        margin=dict(l=10, r=30, t=60, b=10),
        yaxis=dict(title="", automargin=True),
        xaxis=dict(title="Number of Records"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
Beberapa produk skincare muncul berulang kali di katalog karena dapat mengatasi beberapa masalah kulit sekaligus, atau memiliki kombinasi bahan aktif yang beragam. Ini menjadi peluang bagi Soluskin untuk membangun sistem rekomendasi yang tidak hanya terpaku pada satu masalah kulit saja, tetapi juga mempertimbangkan kecocokan antara masalah kulit, bahan aktif, dan produk secara menyeluruh.
"""
    )


def _render_ingredient_by_problem(df: pd.DataFrame) -> None:
    st.subheader("Bahan Aktif Teratas untuk Setiap Masalah Kulit")

    problem_ingredient = pd.crosstab(df["problem"], df["ingredient"])

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_problem = st.selectbox(
            "Pilih masalah kulit:", sorted(problem_ingredient.index.tolist())
        )
    with col2:
        top_n = st.slider("Jumlah bahan teratas:", min_value=5, max_value=20, value=10)

    top_for_problem = (
        problem_ingredient.loc[selected_problem]
        .sort_values(ascending=False)
        .head(top_n)
        .to_frame("record_count")
    )
    st.dataframe(top_for_problem, width="stretch")

    st.markdown(
        """
Setiap permasalahan kulit memiliki karakteristik bahan aktif yang berbeda, namun beberapa bahan aktif ternyata dapat dikaitkan dengan lebih dari satu permasalahan kulit sekaligus. Ini menunjukkan bahwa rekomendasi skincare dapat dibangun melalui hubungan antara **kondisi kulit → bahan aktif → produk** — sejalan dengan konsep Soluskin: mengidentifikasi kondisi kulit terlebih dahulu, menentukan bahan aktif yang sesuai, baru kemudian memberikan rekomendasi produk.
"""
    )


def _render_conclusion() -> None:
    st.markdown("---")
    st.subheader("Kesimpulan")
    st.markdown(
        """
- Data katalog produk sudah bersih (tanpa *missing values* atau duplikat) dan mencakup jumlah data yang memadai untuk kelima masalah kulit yang didukung.
- Karena seluruh data berasal dari satu retailer dan belum memiliki kolom harga/rating, rekomendasi saat ini bersifat *"cocok atau tidak cocok"* berdasarkan kesesuaian bahan aktif, bukan peringkat berdasarkan popularitas atau harga.
- Beberapa bahan aktif dan produk bersifat lintas-masalah kulit — ini adalah dasar dari mekanisme pencocokan **kondisi kulit → bahan aktif → produk** yang digunakan pada halaman *Prediction*.
"""
    )


def run() -> None:
    st.title("Analisis Eksplorasi Data Katalog Produk Skincare")
    st.markdown(
        """
### Mempersiapkan Data untuk Rekomendasi Skincare Berbasis Permasalahan Kulit

Sebelum sistem merekomendasikan produk berdasarkan *masalah kulit* yang terdeteksi (lihat halaman **Prediction**), ada baiknya kita memahami dulu data yang menjadi dasar rekomendasi tersebut: seberapa seimbang katalog di setiap masalah kulit, bahan aktif apa yang paling berperan, dan di bagian mana data ini masih memiliki keterbatasan.
"""
    )

    df = load_data()

    _render_overview(df)
    _render_data_quality(df)
    _render_problem_distribution(df)
    _render_top_ingredients(df)
    _render_top_products(df)
    _render_ingredient_by_problem(df)
    _render_conclusion()


if __name__ == "__main__":
    run()
