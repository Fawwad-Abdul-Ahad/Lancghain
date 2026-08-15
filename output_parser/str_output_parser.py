from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Who is the Prime Minister of Pakistan?")

template1 = PromptTemplate(
    template="Write a detal article on this {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = "Write a five line summary on the following text /n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'Attention'})

print(result)