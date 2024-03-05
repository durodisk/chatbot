import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader('Texts', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#saving the vector
persist_dir = 'MyVectorEmbeddings'
doc_vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings, 
    persist_directory=persist_dir)

doc_vectorstore.persist()
