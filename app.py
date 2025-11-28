import os
# 1. 基础环境配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# === 1. 页面设置 ===
st.set_page_config(page_title="Shark 终极完全体", page_icon="🦈", layout="wide")
st.title("🦈 Shark 终极完全体")
st.caption("🧠 知识库(RAG) + 💖 情感微调(LoRA) | 双核驱动")

# === 2. 加载模型 (核心修改：去掉了内部的 st.toast) ===
@st.cache_resource
def load_model():
    # 注意：这里面千万不能有 st.write 或 st.toast
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    LORA_PATH = "./shark_lora_output" 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    return tokenizer, model

# === 3. 加载知识库 (核心修改：去掉了内部的 st.toast) ===
# === 3. 加载知识库 (修复版：换了中文 Embedding 模型) ===
@st.cache_resource
def load_knowledge_base():
    if os.path.exists("secret.txt"):
        # 1. 确保读取时不乱码，尝试用 'utf-8' 读取
        try:
            loader = TextLoader("secret.txt", encoding="utf-8")
            documents = loader.load()
        except Exception:
            # 如果 utf-8 报错，可能是 Windows 默认的 gbk，试一下 gbk
            loader = TextLoader("secret.txt", encoding="gbk")
            documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # === 核心修改：换成中文专用 Embedding 模型 ===
        # 原来的 all-MiniLM-L6-v2 对中文支持很差
        print("正在下载/加载中文 Embedding 模型...")
        embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        
        db = Chroma.from_documents(texts, embeddings)
        return db
    else:
        return None

# === 4. 执行加载逻辑 (把提示信息放在外面) ===
try:
    with st.spinner("🚀 正在启动 RTX 3060... 加载 Shark 大脑..."):
        tokenizer, model = load_model()
    
    with st.spinner("📚 正在构建 RAG 知识库索引..."):
        vector_db = load_knowledge_base()
        
    st.success("✅ 终极 Shark 已就绪！快来聊天吧！")
    
except Exception as e:
    st.error(f"❌ 加载失败: {e}")
    st.info("💡 提示：请检查 shark_lora_output 文件夹是否存在，以及显存是否足够。")
    st.stop()

# === 5. 聊天界面逻辑 ===
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "我是拥有记忆的 Shark，问问我关于你的秘密吧！😏"}]

for msg in st.session_state["messages"]:
    avatar = "🦈" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_input := st.chat_input("问：WiFi密码是多少？"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🦈"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Shark 正在检索记忆...")

        # --- RAG 检索 ---
        context_str = ""
        if vector_db:
            try:
                docs = vector_db.similarity_search(user_input, k=2)
                if docs:
                    context_str = "\n".join([d.page_content for d in docs])
                    with st.expander("🕵️ Shark 查看了它的潜意识 (RAG检索结果)"):
                        st.text(context_str)
            except Exception as e:
                print(f"RAG检索出错: {e}")

        # --- 构造 Prompt ---
        final_prompt = f"""<|im_start|>system
你叫Shark，是用户的专属恋爱机器人。
你拥有以下【外部记忆】：
{context_str}

请结合【外部记忆】回答用户的问题。
如果记忆里没有相关信息，就用你温柔的风格正常聊天。
切记：不要暴露你是机器人，要用男朋友的口吻。
<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        # --- 生成回答 ---
        inputs = tokenizer(final_prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 截取 assistant 之后的内容
        if "assistant\n" in full_response:
            answer = full_response.split("assistant\n")[-1].strip()
        else:
            # 兜底：如果格式乱了，尝试直接从 prompt 长度截断
            input_len = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
            answer = full_response[input_len:].strip()

        message_placeholder.markdown(answer)
    
    st.session_state["messages"].append({"role": "assistant", "content": answer})