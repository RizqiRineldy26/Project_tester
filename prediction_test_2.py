import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import os
import nltk

from supabase import create_client
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import NLTKTextSplitter
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")


def checkpoint(name):
    print(f"CHECKPOINT: {name}", flush=True)


CHAT_MODEL = "models/gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
CHROMA_DIR = "./chroma_db_gemini_3072"
JOURNAL_BUCKET = "Journal"
JOURNAL_FILE = "jurnal skincare.pdf"

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))

checkpoint("before chat_model")
chat_model = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=CHAT_MODEL)
checkpoint("after chat_model")

checkpoint("before supabase client")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
checkpoint("after supabase client")


@st.cache_data
def load_table():
    checkpoint("load_table: start")
    all_rows = []
    batch_size = 1000
    start = 0

    while True:
        response = (
            supabase.table("skincare_cleaned")
            .select("*")
            .range(start, start + batch_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        all_rows.extend(batch)
        start += batch_size

    checkpoint(f"load_table: done, {len(all_rows)} rows")
    return pd.DataFrame(all_rows)


checkpoint("before load_table()")
df = load_table()
checkpoint("after load_table()")


def _ensure_nltk_punkt():
    checkpoint("nltk: checking punkt_tab")
    try:
        nltk.data.find("tokenizers/punkt_tab")
        checkpoint("nltk: punkt_tab already present")
    except LookupError:
        checkpoint("nltk: downloading punkt_tab")
        nltk.download("punkt_tab")
        checkpoint("nltk: punkt_tab downloaded")


@st.cache_resource
def build_retriever():
    checkpoint("build_retriever: start")

    checkpoint("build_retriever: creating embedding_model")
    embedding_model = GoogleGenerativeAIEmbeddings(
        google_api_key=GEMINI_API_KEY,
        model=EMBEDDING_MODEL,
    )
    checkpoint("build_retriever: embedding_model created")

    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        checkpoint("build_retriever: loading existing Chroma DB")
        db_connection = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model,
        )
        checkpoint("build_retriever: existing Chroma DB loaded")
        return db_connection.as_retriever(search_kwargs={"k": 10})

    checkpoint("build_retriever: getting pdf public url")
    pdf_url = supabase.storage.from_(JOURNAL_BUCKET).get_public_url(JOURNAL_FILE)
    checkpoint(f"build_retriever: got pdf url ({pdf_url})")

    checkpoint("build_retriever: creating PyPDFLoader")
    loader = PyPDFLoader(pdf_url)
    checkpoint("build_retriever: PyPDFLoader created, calling load_and_split")
    pages = loader.load_and_split()
    checkpoint(f"build_retriever: load_and_split done, {len(pages)} pages")

    _ensure_nltk_punkt()

    checkpoint("build_retriever: creating text splitter")
    text_splitter = NLTKTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    checkpoint("build_retriever: splitting documents")
    chunks = text_splitter.split_documents(pages)
    checkpoint(f"build_retriever: split done, {len(chunks)} chunks")

    checkpoint("build_retriever: calling Chroma.from_documents (embedding step)")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    checkpoint("build_retriever: Chroma.from_documents done")

    return db.as_retriever(search_kwargs={"k": 10})


checkpoint("before build_retriever()")
with st.spinner("Loading the resource..."):
    retriever = build_retriever()
checkpoint("after build_retriever()")


chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            You are an AI that gives skincare recommendations based on the provided context.
            Only recommend products and ingredients that appear in the "Matching products" list below.
            Do not invent or assume ingredients that aren't listed there.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Give your recommended ingredients first, then give your recommended product from the matching product list that match the ingredients.

            journal context: {context}
            matching product: {product_context}
            skin condition: {skin_condition}

            INSTRUCTIONS FOR PRODUCT MATCHING:
            - DO NOT require an exact ingredient match (e.g., if Isotretinoin is mentioned, do not restrict yourself only to products explicitly named 'Isotretinoin').
            - Match products based on SEMANTIC SIMILARITY, intended use, and shared benefits:
            - Compare the product descriptions, target concerns, and active categories (e.g., exfoliants, spot treatments, retinoids, soothing agents).
            - If a exact active ingredient is absent, recommend products from the list that address the user's overall skin condition ({skin_condition}).
            - Clearly explain WHY each product is recommended (e.g., "While this product does not contain Isotretinoin, it contains BHA which addresses similar pore congestion...").
            - Only output "None" if the provided product list is completely empty or completely irrelevant to skincare.
            - Show preview link that shows the image of the product recomendation.
            - Generate the image to show the image base on product link.
            """
        ),
    ]
)

output_parser = StrOutputParser()


def filter_product(skin_conditions):
    filtered = df[df['problem'].isin(skin_conditions)]
    return filtered


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": (lambda x: x['skin_conditions_str']) | retriever | format_docs,
     "product_context": lambda x: filter_product(x['skin_conditions_list']),
     "skin_condition": lambda x: x['skin_conditions_str']}
    | chat_template
    | chat_model
    | output_parser
)


@st.cache_resource
def load_classifier():
    checkpoint("load_classifier: start")
    m = tf.keras.models.load_model('model_2_sigmoid.keras')
    checkpoint("load_classifier: done")
    return m


checkpoint("before load_classifier()")
model = load_classifier()
checkpoint("after load_classifier()")

checkpoint("MODULE LOAD COMPLETE")


def run():
    st.title("Check your skin condition!")

    enable = st.checkbox('Enable your camera')
    camera = st.camera_input('Take a picture of your face', disabled=not enable)
    upload = st.file_uploader('Choose a file')

    picture = camera if camera is not None else upload

    if picture is not None:
        bytes_data = picture.getvalue()
        img_tensor = tf.io.decode_image(bytes_data, channels=3)
        img_tensor = tf.image.resize(img_tensor, [150, 150])
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        pred_prob = model.predict(img_tensor)
        pred_class = np.argmax(pred_prob[0])
        class_names = ['acne', 'blackheads', 'dark spots', 'pores', 'wrinkles']

        pred_class_name = class_names[pred_class]
        pct_prob = [f'{x * 100:.2f}%' for x in pred_prob[0]]

        inf_df = pd.DataFrame({
            'skin_problem': class_names,
            'prediction': pct_prob
        })

        st.write('## The number one concern from your face is:', pred_class_name)
        st.write('#### Scroll down to check the full prediction!')
        st.image(picture)
        st.dataframe(inf_df)

        inf_df['prediction'] = (inf_df['prediction'].astype(str).str.strip('%').astype(float))
        skin_conditions_list = inf_df.loc[inf_df['prediction'] > 30, 'skin_problem'].tolist()
        skin_conditions_str = ", ".join(skin_conditions_list)

        st.title("Here is your Skin Care Recommendation!")
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({
                "skin_conditions_str": skin_conditions_str,
                "skin_conditions_list": skin_conditions_list,
            })
            st.markdown(response)


if __name__ == '__main__':
    run()