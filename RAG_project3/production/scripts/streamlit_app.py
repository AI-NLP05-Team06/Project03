# 질의분석 포함 전체 파이프라인(generation/compound_answer.py::answer_query)을
# 브라우저에서 채팅창으로 테스트하기 위한 간단한 Streamlit 앱입니다.
# 실행: streamlit run production/scripts/streamlit_app.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from generation.compound_answer import answer_query

st.set_page_config(page_title="예금보험공사 RAG 챗봇 (테스트용)")
st.title("예금보험공사 RAG 챗봇 (테스트용)")

if "history" not in st.session_state:
    st.session_state.history = []
if "previous_question" not in st.session_state:
    st.session_state.previous_question = None

if st.button("맥락 초기화"):
    st.session_state.history = []
    st.session_state.previous_question = None
    st.rerun()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])

question = st.chat_input("질문을 입력하세요")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            answer = answer_query(question, previous_question=st.session_state.previous_question)
        st.write(answer)

    st.session_state.history.append({"question": question, "answer": answer})
    st.session_state.previous_question = question
