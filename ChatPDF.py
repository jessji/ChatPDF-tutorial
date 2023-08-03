import os
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY # 填入你的OpenAI API Key，或者在命令行输入“export OPENAI_API_KEY='sk-...'”，将其设置为环境变量

from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
# import pinecone

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import streamlit as st

# 加载文档并分割成小段文本texts (chunks)
def load_and_split(path: str):
    # 加载pdf文档
    loader = PyPDFLoader(path)
    # loader = UnstructuredPDFLoader(path)
    # loader = OnlinePDFLoader("https://www.goldmansachs.com/intelligence/pages/top-of-mind/generative-ai-hype-or-truly-transformative/report.pdf")
    docs = loader.load() # 用pypdf load文档时，默认按照分页来拆分文档
    print(f'你的文档被拆分成了{len(docs)}份，第一份有{len(docs[0].page_content)}个字\n')

    # 将文档（进一步）拆分成小段文本texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    print(f'你的文档被（进一步）拆分成了{len(texts)}份，第一份有{len(texts[0].page_content)}个字\n')

    return texts


##############################
# 第一部分：向量数据库操作
##############################

# 从向量数据库向量索引Index，用于用户问答
index_name = "my-knowledgebase"

#################################################
# 情况1：如果还没有建立该知识库的向量索引Index，需新建
#################################################

texts = load_and_split("data/llm-labor-market-impact-sample.pdf")

embeddings = OpenAIEmbeddings()

# 方法1: 用Chroma向量数据库
persist_directory = "db"
vectordb = Chroma.from_texts(documents=texts, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()

# # 方法2: 用Pinecone向量数据库
# pinecone.init(
#     api_key=PINECONE_API_KEY, # 在 app.pinecone.io 的“API Keys”页面查看
#     environment=PINECONE_API_ENV # 在api key旁边
# )
# if index_name not in pinecone.list_indexes():
#     # we create a new index
#     pinecone.create_index(
#       name=index_name,
#       metric='cosine',
#       dimension=1536
# )
# # 将texts转化为向量(vectors)并存入pinecone Index中
# vectordb = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name) # 填入你的pinecone index名称
# print("***已完成vectors更新插入(upsert)***")

###################################
# 情况2：如果已经建立向量索引，则直接加载
###################################

# # 从Pinecone向量数据库加载向量索引Index
# pinecone.init(
#     api_key=PINECONE_API_KEY, # 在 app.pinecone.io 的“API Keys”页面查看
#     environment=PINECONE_API_ENV # 在api key旁边
# )
# vectordb = Pinecone.from_existing_index(index_name, OpenAIEmbeddings())

######################################
# 情况3：添加新的数据 (texts)至已有的Index
######################################

# # 添加新的texts到Pinecone向量数据库
# new_texts = ""
# index = pinecone.Index(index_name)
# # 初始化pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY, # 在 app.pinecone.io 的“API Keys”页面查看
#     environment=PINECONE_API_ENV # 在api key旁边
# )
# vectordb = Pinecone(index, OpenAIEmbeddings().embed_query, "text")
# vectordb.add_texts(new_texts)


#################
# 第二部分：用户问答
#################

# 用streamlit生成web界面
st.title('ChatPDF')
user_input = st.text_input('请输入您的问题')

# 根据用户数据，生成回复
if user_input:
    print(f"用户输入：{user_input}")
    # 根据用户输入，从向量数据库搜索得到相似度最高的texts
    # most_relevant_texts = vectordb.similarity_search(user_input, k=2) # k是返回的texts数量，默认为4

    # 搜索得到与用户输入相似度最大、而彼此之间有差异的texts
    # k是返回的texts数量，默认为4
    # fetch_k为输入给MMR算法的texts数量（用来得到k个texts），默认为20
    # lambda_mult为返回texts之间差异性，1为最大，0为最小，默认为0.5
    most_relevant_texts = vectordb.max_marginal_relevance_search(user_input, k=2, fetch_k=6, lambda_mult=1)
    # print("以下是相关度最高的段落节选：")
    # print(most_relevant_texts[0].page_content[:450])

    # chain_type: stuff（不分段）, map_reduce（分段、分别请求）, refine（分段、依次请求并优化结果，比前者慢）, map-rerank（分段请求，根据分数返回结果）
    llm = OpenAI(temperature=0.5)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=most_relevant_texts, question=user_input+"用中文回答")

    st.write(answer)

    texts_length = 0
    st.write("==============================以下为测试打印数据，可忽略==============================")
    st.write(f"以下是相关度最高的【{len(most_relevant_texts)}】段文本：")
    i = 0
    for t in most_relevant_texts:
        i += 1
        st.write(f"********第【{i}】段********")
        st.write(t.page_content)
        texts_length += len(t.page_content)
    print(f"请求字段长度为：{texts_length}")
