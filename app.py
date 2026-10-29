import streamlit as st
from openai import OpenAI
import numpy as np
import json
import requests
import pandas as pd

# ================= 配置区 =================
# ⚠️ 请填入你的真实 API Key (填完记得保存)
DEEPSEEK_API_KEY = ""
BIGMODEL_API_KEY = "" 
# ==========================================

st.set_page_config(page_title="应急语言服务综合平台", page_icon="🌐", layout="wide")
st.title("🌐 跨国救援应急语言服务综合平台")
st.markdown("欢迎使用！本平台包含：信息聚合、多语种翻译、术语管理、平行语料及 RAG 智能问答。")

ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {"Authorization": f"Bearer {BIGMODEL_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "embedding-3", "input": text}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@st.cache_data
def load_json(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

st.sidebar.header("功能导航")
menu = st.sidebar.radio("请选择服务模块", [
    "1. 外语信息聚合 (新闻)", 
    "2. 多语种翻译", 
    "3. 术语管理", 
    "4. 平行语料库", 
    "5. 智能问答 (RAG)"
])

if menu == "1. 外语信息聚合 (新闻)":
    st.header("📰 多源国际救援新闻中文版")
    news_data = load_json("news.json")
    for news in news_data[:20]: 
        with st.expander(f"[{news['source_site']}] {news['title_zh']} ({news['publish_date']})"):
            st.write(news['content_zh'])
            st.caption(f"原文链接：{news['source_url']} | 状态：{news['status']}")

elif menu == "2. 多语种翻译":
    st.header("📝 东盟多语种应急翻译")
    languages = ["中文", "英语", "越南语", "泰语", "缅甸语", "高棉语", "老挝语", "马来语", "印尼语"]
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("源语言", languages, index=0)
    with col2:
        target_lang = st.selectbox("目标语言", languages, index=2)
        
    text_input = st.text_area("请输入需要翻译的应急文本：", height=100)
    
    if st.button("开始翻译", type="primary"):
        if text_input:
            terms_data = load_json("terms.json")
            sample_terms = "\n".join([f"{t['term_zh']}: {t['term_en']}" for t in terms_data[:5]])
            prompt = f"你是一位专业的应急语言翻译引擎。请将以下{source_lang}文本翻译为{target_lang}。请确保术语准确（参考术语：{sample_terms}）。只输出译文。\n原文：{text_input}"
            
            response = ds_client.chat.completions.create(
                model="deepseek-chat", messages=[{"role": "user", "content": prompt}], stream=True
            )
            result_text = ""
            placeholder = st.empty()
            for chunk in response:
                if chunk.choices[0].delta.content:
                    result_text += chunk.choices[0].delta.content
                    placeholder.markdown(f"**{result_text}**")
            st.success("翻译完成！已自动应用相关术语。")
        else:
            st.warning("请输入文本！")

elif menu == "3. 术语管理":
    st.header("📖 应急术语库管理")
    terms_data = load_json("terms.json")
    df = pd.DataFrame(terms_data)
    
    search_kw = st.text_input("请输入中文或英文关键词检索：")
    if search_kw:
        filtered_df = df[df['term_zh'].str.contains(search_kw, case=False) | df['term_en'].str.contains(search_kw, case=False)]
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(df.head(50), use_container_width=True) 
        st.caption(f"当前术语库总量：{len(df)} 条")

elif menu == "4. 平行语料库":
    st.header("📚 双语平行语料检索")
    corpus_data = load_json("corpus.json")
    df = pd.DataFrame(corpus_data)
    
    search_kw = st.text_input("请输入原文或译文关键词检索：")
    if search_kw:
        filtered_df = df[df['source_zh'].str.contains(search_kw, case=False) | df['target_en'].str.contains(search_kw, case=False)]
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"当前语料库总量：{len(df)} 句")

elif menu == "5. 智能问答 (RAG)":
    st.header("🤖 知识库增强智能问答 (带来源追溯)")
    
    @st.cache_resource
    def build_knowledge_base():
        kb = []
        for n in load_json("news.json"):
            kb.append({"text": n["title_zh"] + n["content_zh"], "source": f"新闻来源: {n['source_url']}"})
        for c in load_json("corpus.json"):
            kb.append({"text": c["source_zh"] + " " + c["target_en"], "source": f"语料库: {c['source']}"})
        for t in load_json("terms.json"):
            kb.append({"text": t["term_zh"] + " 的英文是 " + t["term_en"], "source": f"术语库: {t['source']}"})
            
        # 为了速度，我们只向量化前100条，真正部署时可以全部向量化并存入数据库
        for item in kb[:100]:
            item["vector"] = get_embedding(item["text"])
        return kb[:100]

    user_query = st.text_input("请输入你的问题：")
    if st.button("向 AI 提问", type="primary"):
        if user_query:
            with st.spinner("正在检索知识库..."):
                try:
                    kb = build_knowledge_base()
                    query_vector = get_embedding(user_query)
                    
                    for item in kb:
                        item["score"] = cosine_similarity(query_vector, item["vector"])
                    kb_sorted = sorted(kb, key=lambda x: x["score"], reverse=True)
                    
                    top_k = kb_sorted[:2]
                    context = "\n".join([f"[{i+1}] {item['text']}" for i, item in enumerate(top_k)])
                    
                    prompt = f"请基于以下参考资料回答问题。如果资料中没有答案，请说“未找到”。\n\n资料：\n{context}\n\n问题：{user_query}"
                    
                    response = ds_client.chat.completions.create(
                        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], stream=True
                    )
                    
                    st.markdown("### 💡 AI 回答：")
                    result_text = ""
                    placeholder = st.empty()
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            result_text += chunk.choices[0].delta.content
                            placeholder.markdown(f"{result_text}")
                    
                    st.markdown("### 📚 参考来源：")
                    for i, item in enumerate(top_k):
                        st.write(f"**[{i+1}]** {item['source']} (相关度: {item['score']:.4f})")
                except Exception as e:
                    st.error(f"发生错误：{e}")
        else:
            st.warning("请输入问题！")

