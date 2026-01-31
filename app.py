from text_summarization import preprocess_text , score_sentences , calculate_tfidf , generate_summary

import streamlit as st

st.title("CONCISE")

text= st.text_area("Enter the text you want to summarize ")

if st.button("Submit"):
    try:
        st.write("Preprocessing....")
        sentences= preprocess_text(text)
        st.write("Calculating TF_IDF scores....")
        tf_idf= calculate_tfidf(sentences)
        st.write("Scoring sentences....")
        sentence_score= score_sentences(tf_idf)
        summary= generate_summary(sentences , sentence_score)
        st.subheader('Original text :')
        st.markdown(text)
        st.subheader('Generated summary :')
        st.success(summary)
    except:
        st.error("Some error occured ")

