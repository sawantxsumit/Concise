from text_summarization import preprocess_text , score_sentences , calculate_tfidf , generate_summary

import streamlit as st
from nltk import sent_tokenize
st.title("CONCISE")

text= st.text_area("Enter the text you want to summarize ")

if st.button("Submit"):
    try:
        st.write("Preprocessing....")
        sentence= preprocess_text(text)
        st.write("Calculating TF_IDF scores....")
        tf_idf= calculate_tfidf(sentence)
        st.write("Scoring sentences....")
        sentence_score= score_sentences(tf_idf)
        summary= generate_summary(sentence , sentence_score)
        st.subheader('📌 Original text :')
        st.markdown(text)
        summary_text = summary

        sentences = sent_tokenize(summary_text)

        st.subheader("📌 Generated Summary")

        for sent in sentences:
            st.markdown(f"-> {sent}")
        # st.success(summary)
    except:
        st.error("Some error occured ")

