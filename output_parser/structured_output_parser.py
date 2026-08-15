from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(
        name = "name", description='Write the name'
    ),
    ResponseSchema(name="age", description='Write the age of the person in integer'),
    ResponseSchema(name='city', description= 'Write the city of the person')
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Write the name age city of the fake person /n {formatted_instructions}",
    partial_variables={'formatted_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)
