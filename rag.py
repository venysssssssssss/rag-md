import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Carregar os dados e criar embeddings e vector store
    logger.info("Carregando os dados...")
    loader = WebBaseLoader(
        web_paths=("http://localhost/server.py",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    logger.info("Dados carregados com sucesso.")

    logger.info("Criando embeddings e vector store...")
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    logger.info("Embeddings e vector store criados com sucesso.")

    # Chamar o modelo Ollama Llama3
    def ollama_llm(question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']

    retriever = vectorstore.as_retriever()

    # Uso da aplicação RAG
    logger.info("Chamando a aplicação RAG...")
    formatted_context = "\n\n".join(doc.page_content for doc in docs)
    result = ollama_llm("Como posso modificar este arquivo para servir conteúdo específico de um diretório diferente?", formatted_context)
    logger.info("Resultado obtido com sucesso.")
    print(result)

except Exception as e:
    logger.error(f"Ocorreu um erro: {str(e)}")
