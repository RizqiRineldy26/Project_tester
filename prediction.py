import hashlib
 
import keras
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
 
from rag import fetch_preview_image, recommend_products, stream_answer
 
CLASS_NAMES = ['acne', 'blackheads', 'dark_spots', 'pores', 'wrinkles']
 
 
# Wrapper untuk membuang argumen yang tidak didukung
class SafeDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
 
 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'model_2_sigmoid.keras',
        custom_objects={'Dense': SafeDense}
    )
 
 
def show_ingredient_recommendation(scores, api_key, bytes_data):
    """Tampilkan rekomendasi ingredient dari jurnal.
    Return True jika jawaban berhasil ditampilkan."""
    st.write('### Skincare Ingredient Recommended')
 
    if not api_key:
        st.info('Insert API KEY di sidebar untuk melihat rekomendasi ingredient dan produk.')
        return False
 
    img_hash = hashlib.md5(bytes_data).hexdigest()
 
    # Jawaban untuk gambar yang sama sudah ada -> tampilkan ulang tanpa memanggil LLM
    if st.session_state.get('img_hash') == img_hash:
        st.markdown(st.session_state['answer'])
        return True
 
    try:
        answer = st.write_stream(stream_answer(scores, api_key))
        st.session_state['img_hash'] = img_hash
        st.session_state['answer'] = answer
        return True
    except Exception as e:
        st.error(f'Gagal memanggil LLM: {e}')
        return False
 
 
def show_product_cards(products, cols, per_row=3):
    """Tampilkan produk sebagai kartu: gambar, nama, brand, harga, link."""
    name_col = cols.get('name') or products.columns[1]
    for start in range(0, len(products), per_row):
        row = products.iloc[start:start + per_row]
        for column, (_, item) in zip(st.columns(per_row), row.iterrows()):
            with column:
                link = str(item[cols['link']]) if cols.get('link') else None
 
                # gambar: pakai kolom gambar bila ada, kalau tidak ambil dari halaman produk
                img = str(item[cols['image']]) if cols.get('image') else None
                if not img and link and link.startswith('http'):
                    img = fetch_preview_image(link)
                if img and img.startswith('http'):
                    st.image(img, use_container_width=True)
                else:
                    st.caption('(gambar tidak tersedia)')
 
                st.markdown(f"**{item[name_col]}**")
                if cols.get('brand'):
                    st.caption(item[cols['brand']])
                if cols.get('price'):
                    st.write(item[cols['price']])
                st.caption('Bahan: ' + item['bahan_sesuai_jurnal'])
                if link and link.startswith('http'):
                    st.link_button('Lihat produk', link)
 
 
def show_product_recommendation(scores):
    st.write('### Rekomendasi Produk per Bahan')
    st.caption('Bahan diurutkan dari persentase dermatolog yang merekomendasikan '
               '(Alvarez et al., JAAD 2025). Produk diambil dari database.')
    try:
        recommendations = recommend_products(scores)
        for idx, rec in enumerate(recommendations, start=1):
            label = f"{idx}. {rec['ingredient'].title()} — direkomendasikan {rec['pct']}% dermatolog"
            with st.expander(label, expanded=(idx == 1)):
                if rec['prescription']:
                    st.info('Bahan ini umumnya hanya tersedia dengan resep dokter.')
                if rec['products'].empty:
                    st.write('Belum ada produk di database yang mengandung bahan ini.')
                else:
                    show_product_cards(rec['products'], rec['cols'])
                    with st.popover('Lihat sebagai tabel'):
                        st.dataframe(rec['products'], hide_index=True)
    except Exception as e:
        st.error(f'Gagal mengambil produk dari Supabase: {e}')
 
 
def run():
    st.title("Check your skin problem!")
 
    api_key = st.sidebar.text_input(
        'Insert API Key',
        type='password',
        help='Dapatkan gratis di aistudio.google.com'
    )
 
    model = load_model()
 
    enable = st.checkbox('Enable your camera')
    camera = st.camera_input('Take a picture of your face', disabled=not enable)
    upload = st.file_uploader('Choose a file', type=['jpg', 'jpeg', 'png'])
 
    picture = camera if camera is not None else upload
    if picture is None:
        return
 
    # Preprocessing
    bytes_data = picture.getvalue()
    img_tensor = tf.io.decode_image(bytes_data, channels=3, expand_animations=False)
    img_tensor = tf.image.resize(img_tensor, [150, 150])
    img_tensor = tf.expand_dims(img_tensor, axis=0)
 
    # Prediksi
    pred_prob = model.predict(img_tensor)
    pred_class_name = CLASS_NAMES[int(np.argmax(pred_prob[0]))]
    scores = {c: float(s) for c, s in zip(CLASS_NAMES, pred_prob[0])}
 
    st.write('## This is the result:', pred_class_name)
    st.write('#### Scroll down to check the full prediction!')
    st.image(picture)
    st.dataframe(pd.DataFrame({
        'skin problem': CLASS_NAMES,
        'prediction': [f'{x * 100:.2f}%' for x in pred_prob[0]],
    }))
 
    # 1) Ingredient dari jurnal dulu
    answer_ready = show_ingredient_recommendation(scores, api_key, bytes_data)
 
    # 2) Produk hanya muncul setelah rekomendasi ingredient berhasil tampil
    if answer_ready:
        st.divider()
        show_product_recommendation(scores)
 
 
if __name__ == '__main__':
    run()