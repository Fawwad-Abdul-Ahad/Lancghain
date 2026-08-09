from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

chat_template = ChatPromptTemplate(
    [
        ('system' , 'You are helpful customer suppert assistant'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{query}')
    ]
)

chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
prompt = chat_template.invoke({'chat_history':chat_history, 'query' : "what is the adsadaas"})
print(prompt)

