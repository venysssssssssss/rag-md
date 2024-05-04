import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os
import markdown

# Carregar, dividir e recuperar documentos
file_path = os.path.join("data", "alice.md")
loader = UnstructuredMarkdownLoader(file_path=(file_path))
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Função para formatar documentos
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Função que define a cadeia RAG
def rag_chain(question):  
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Interface Gradio
iface = gr.Interface(
    fn=rag_chain,
    inputs=["text"],
    outputs="text",
    title="RAG Chain Question Answering",
    description="Enter a query to get answers from the RAG chain."
)

# Inicializar a aplicação
iface.launch()
