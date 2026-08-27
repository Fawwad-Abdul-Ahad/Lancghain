from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough,RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()
model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detail document about the following topic /n {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a summary of the following text /n {text}",
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1, model1, parser)
summarize_chain = (lambda x: {"text": x}) | prompt2 | model1 | parser
conditional_gen_chain = RunnableBranch(
    (lambda x : len(x.split())>300, RunnableSequence(summarize_chain)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, conditional_gen_chain)
print(final_chain.invoke({'topic' : "Russia vs Ukraine"}))