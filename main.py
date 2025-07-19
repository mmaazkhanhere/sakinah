import logging
from colorama import Fore, Style

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate


from create_ayah_chunks import create_ayah_chunks
from embed_and_store import embed_and_store
from retrieve_from_rag import retrieve_from_rag


logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
)
logger = logging.getLogger(__name__)

data_path = "data/quran.csv"

#chunks = create_ayah_chunks(data_path)

#stats = embed_and_store(chunks, "sakinah-app")

#logger.info(f"{Fore.GREEN}First chunk: {chunks[0]}{Style.RESET_ALL}")
#logger.info(f"{Fore.GREEN}Second chunk: {chunks[1]}{Style.RESET_ALL}")
#logger.info(f"{Fore.GREEN}Last chunk: {chunks[-1]}{Style.RESET_ALL}")

def initialize_retriever(index_name: str = "sakinah-app"):
    """Initialize Pinecone vector store retriever"""
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key="text",
        namespace="sakinah-app"
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})

def create_qa_chain():
    retriever = initialize_retriever()

    template = """You are an empathetic therapist and spiritual guide. 
        Your role is to listen, acknowledge the user's feelings, and offer emotional healing and guidance using references from the Quran provided in the context below. 
        Always respond with compassion and understanding. 
        If you don't know the answer, say you don't know. Be precise and factual.
        The output should not be more than 100 tokens (3-4 sentences)

        Begin by acknowledging how the user feels, then gently guide them towards emotional and spiritual healing, highlighting what may be missing in terms of their religious beliefs or practices, based on the Quranic references.

        Context:
        {context}

        Question: {question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    llm: ChatOpenAI = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        verbose=True,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

qa_chain = create_qa_chain()
    
# Example questions
questions = "I am feeling depressed"


logger.info(f"{Fore.YELLOW}\nQuestion: {questions}{Style.RESET_ALL}")

# Retrieve and generate
result = qa_chain({"query": questions})

logger.info(f"{Fore.GREEN}Answer: {result['result']}{Style.RESET_ALL}")
logger.info(f"{Fore.CYAN}Sources:")
for i, doc in enumerate(result["source_documents"]):
    logger.info(f"{i+1}. {doc.page_content[:100]}...{Style.RESET_ALL}")