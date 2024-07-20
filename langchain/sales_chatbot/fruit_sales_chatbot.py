import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10810'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10810'
def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = False

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if len(ans["source_documents"])!=0 or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        chat_model = ChatOpenAI()
        prompt = PromptTemplate.from_template("你是专业的水果销售商，请认真回答user的问题: {question}")
        return chat_model.predict_messages(prompt.format(question=message),temperature=0)


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="水果销售客服",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化水果销售机器人
    initialize_sales_bot(vector_store_dir="D:/githome/ITA/openai-quickstart/langchain/sales_chatbot/real_estates_sale")
    # 启动 Gradio 服务
    launch_gradio()
