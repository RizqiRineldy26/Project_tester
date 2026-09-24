
import streamlit as st

st.set_page_config(
    page_title="Soluskin",
    page_icon="",
    layout="wide",
)

prediction_page = st.Page(
    "prediction_test_2.py",
    title="Skin Prediction & Recommendation",
    default=True,
)

eda_page = st.Page(
    "eda.py",
    title="EDA Katalog Produk",
)

pg = st.navigation([prediction_page, eda_page])
pg.run()
