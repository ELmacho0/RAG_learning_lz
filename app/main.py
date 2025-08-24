import json
import os
from pathlib import Path

import streamlit as st

import rag  # local retrieval utilities

ACCOUNTS_FILE = Path(__file__).resolve().parent.parent / "config" / "accounts.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_accounts():
    if ACCOUNTS_FILE.exists():
        with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {u["username"]: u["password"] for u in data.get("users", [])}
    return {}


def ensure_user_dirs(user: str):
    (DATA_DIR / user / "uploads").mkdir(parents=True, exist_ok=True)


def login_view():
    st.markdown(
        "<div style='background-color:#FDE68A;padding:8px;border-radius:4px;'>仅供演示，不代表生产安全实践</div>",
        unsafe_allow_html=True,
    )
    st.subheader("登录")
    username = st.text_input("账号")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        accounts = load_accounts()
        if accounts.get(username) == password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            ensure_user_dirs(username)
            st.experimental_rerun()
        else:
            st.error("账号或密码错误")


def chat_page():
    st.subheader("知识库智能问答")
    user = st.session_state.get("user")
    st.write(f"当前用户：{user}")
    query = st.text_input("请输入您的问题")
    if st.button("发送"):
        if query.strip():
            try:
                answer = rag.chat_with_docs(user, query)
                st.write(answer)
            except Exception as e:
                st.error(str(e))


def kb_page():
    st.subheader("个人知识库")
    user = st.session_state.get("user")
    uploads_dir = DATA_DIR / user / "uploads"
    uploaded = st.file_uploader("上传文件", accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            dest = uploads_dir / file.name
            with open(dest, "wb") as f:
                f.write(file.getvalue())
            try:
                rag.index_file(user, dest)
            except Exception as e:
                st.error(f"索引 {file.name} 失败: {e}")
        st.success("上传并入库完成")
    if uploads_dir.exists():
        st.write("当前已有文件：")
        for p in uploads_dir.iterdir():
            st.write("- ", p.name)


def main():
    st.set_page_config(page_title="知识库智能问答", layout="wide")
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        login_view()
        return

    st.sidebar.title("导航")
    page = st.sidebar.radio("", ("对话", "知识库"))
    st.sidebar.button("退出登录", on_click=lambda: st.session_state.update({"authenticated": False}))
    if page == "对话":
        chat_page()
    else:
        kb_page()


if __name__ == "__main__":
    main()
