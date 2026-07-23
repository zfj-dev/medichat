import streamlit as st
import requests

st.set_page_config(page_title="MediChat", page_icon="🏥")
st.title("🏥 MediChat —— 医疗科普问答助手")
st.caption("本助手仅提供医学科普，不构成诊断或治疗建议。")

API = "http://localhost:8000/chat"

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if p := st.chat_input("请输入您的健康问题..."):
    st.session_state.msgs.append({"role": "user", "content": p})
    with st.chat_message("user"):
        st.write(p)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                r = requests.post(API, json={"message": p}, timeout=180)
                d = r.json()
                a = d.get("answer", "服务异常")
                s = d.get("sources", [])
            except Exception as e:
                a = f"请求出错：{e}"
                s = []
            st.write(a)
            if s:
                st.divider()
                st.caption("📚 参考资料：")
                for x in s[:3]:
                    st.caption(f"· {x['title']}（{x['source']}）")
    st.session_state.msgs.append({"role": "assistant", "content": a})
