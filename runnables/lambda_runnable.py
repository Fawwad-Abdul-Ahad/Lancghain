from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()
model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke aout the following topic /n {topic}",
    input_variables=["topic"]
)
def count(words):
    a = words.split()
    return len(a)

joke = RunnableSequence(prompt1, model1, parser)

parallel_runnnable = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(count)
})

seq_runnable= RunnableSequence(joke, parallel_runnnable)

final_chain = seq_runnable.invoke({'topic': "AI"})
print(final_chain)
