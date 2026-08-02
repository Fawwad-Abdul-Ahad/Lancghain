from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Who is the Prime Minister of Pakistan?")

print(response.content)