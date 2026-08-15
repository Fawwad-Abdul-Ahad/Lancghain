from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


parser = JsonOutputParser()

template1 = PromptTemplate(
    template="Write a 5 lines about the following topic {topic}/n {json_format}",
    input_variables=['topic'],
    partial_variables={'json_format': parser.get_format_instructions()}
)

chain = template1 | model | parser

response = chain.invoke({'topic' : "Attention"})

print(response)