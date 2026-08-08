from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# dynamic prompts
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

domain = input("domain : ")
query = input("query : ")
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} assistant.'),
    ('human', 'Tell me about {query}')
]) 
formatted_prompt = chat_template.invoke({'domain': domain, 'query': query})
res = model.invoke(formatted_prompt)
print(res.content)      