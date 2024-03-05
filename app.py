__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.title('🦜🔗 Golden FAQ AI chatbot')

# persist embeddings
# persist_dir = 'MyVectorEmbeddings'
# vectordb = Chroma(persist_directory=persist_dir , embedding_function=OpenAIEmbeddings())

# non-persist embeddings
text_loader_kwargs = {'autodetect_encoding': True}
loader = DirectoryLoader('Texts', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0, max_tokens=300)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)

def generate_response(query):
    return qa_chain.run(query)

with st.form('my_form'):
    text = st.text_area('Enter text:', 'Hi')
    submitted = st.form_submit_button('Submit')
    if submitted:
        response = generate_response(text)
        if response is not None:
            st.info(response)
