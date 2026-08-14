import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
history_message = [
    SystemMessage(content = "You are a helpful AI Assistant")
]
# Bare string format "gemini-1.5-pro" ya "gemini-1.5-flash"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)
while True:
    u_input = input("You : ")
    history_message.append(HumanMessage (content = u_input))
    if u_input == "exit":
        break
    res = llm.invoke(history_message)
    history_message.append(res)
    print(res.content)
print(history_message)