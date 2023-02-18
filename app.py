import streamlit as st
import pandas as pd
from transformers import pipeline
st.set_page_config(layout="wide")



def initialize_model():
    global qa_model
    qa_model = pipeline("question-answering")
    return qa_model
qa_model = initialize_model()
df = pd.read_csv("originial.csv")

options = sorted(df["Name"].tolist())
default_ix = options.index(1)
tmp_index = "1"

# Seçenekleri içeren liste
index_article = st.sidebar.selectbox("Select a article number", options,index=default_ix)



col1, col2 = st.columns([2,1])

with col1:
    # Başlık
    st.title(df[df["Name"]==index_article]["Title"].values[0])

    f = open("articles/" + str(index_article) + ".txt", "r", encoding="utf-8")
    text = f.read()
    f.close()

    # Full text başlığı ve paragrafı
    st.header("Full text")
    st.write(text)

with col2:
    # Abstract başlığı ve paragrafı
    st.header("Summary")
    st.write(df[df["Name"]==index_article]["Summary"].values[0])

    keyword_text = f"{df[df['Name']==index_article]['K1'].values[0]}, {df[df['Name']==index_article]['K2'].values[0]}, {df[df['Name']==index_article]['K3'].values[0]}, {df[df['Name']==index_article]['K4'].values[0]}, {df[df['Name']==index_article]['K5'].values[0]}"
    st.header("Keyword")
    st.write(keyword_text)
    
    with st.form("QA",clear_on_submit=True):
        search_text = st.text_input('Ask a question', '')
        submitted = st.form_submit_button("Submit")
    if submitted :
        st.write("Your question is:", search_text)
        print("loading...")
        answer = qa_model(question=search_text, context=text)

        st.write('The answer is', answer)
        print("done.")
    
