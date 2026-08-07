from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

docs = [
    "Artificial Intelligence is changing the way people work.",
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand human language.",
    "Large language models can answer questions and generate text.",
    "LangChain is a framework for building LLM applications.",
    "RAG combines retrieval with language models for better answers.",
    "Embeddings convert text into numerical vectors.",
    "Vector databases store embeddings for semantic search.",
    "FAISS is a popular vector database for similarity search.",
    "Chroma is another vector database used in RAG systems.",
    "Python is the most popular language for AI development.",
    "PyTorch is widely used to train deep learning models.",
    "TensorFlow is another framework for machine learning.",
    "Hugging Face provides open-source AI models and datasets.",
    "Ollama allows you to run large language models locally.",
    "Semantic search finds documents based on meaning instead of keywords."
]
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=200)
query = "tell me about machine learning and deep learning"
doc = embedding.embed_documents(docs)
query = embedding.embed_query(query)

score = (cosine_similarity([query], doc))[0]
print(score)
index, score = sorted(list(enumerate(score)),key= lambda x:x[1])[-1]

print(docs[index])
print(score)