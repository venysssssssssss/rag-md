import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os
import torch

# Verificar se há disponibilidade de GPU e configurar o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Função para carregar, dividir e recuperar documentos
def load_and_retrieve_docs(file_path):  
    loader = UnstructuredMarkdownLoader(file_path=(file_path))
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Função para formatar documentos
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Função que define a cadeia RAG
def rag_chain(question):  
    file_path = os.path.join("data", "alice.md")
    retriever = load_and_retrieve_docs(file_path)  
    retrieved_docs = retriever.invoke(question, timeout=30)  # Limitando o tempo de resposta do Ollama
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}], timeout=30, device=device)
    return response['message']['content']

# Interface Gradio
iface = gr.Interface(
    fn=rag_chain,
    inputs="text",
    outputs="text",
    title="RAG Chain Question Answering",
    description="Enter a query to get answers from the RAG chain based on the document 'alice.md'."
)

# Inicializar a aplicação
iface.launch()
