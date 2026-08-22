from typing import Literal

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import Field, BaseModel

from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

class Feedback(BaseModel):
    feedback_text : Literal['positive', 'negative'] = Field(description="Give the positive or negative review on feedback")

parser1= StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Ive me a sentiment review about the following fedback /n {feedback} {formatted_instructions}",
    input_variables=['feedback'],
    partial_variables={"formatted_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Give an appropriate response to this positive feedback like professional way /n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = 'Give an appropriate response to this negtive feedback like professional way /n {feedback}',
    input_variables=['feedback']
)

result = classifier_chain.invoke({'feedback': 'what a wonderful product really nice but not good as the other competitors'})

branch_chain = RunnableBranch(
    (lambda x:x.feedback_text == 'positive', prompt2 | model | parser1),
    (lambda x:x.feedback_text == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find the sentiment")
)

chain = classifier_chain | branch_chain
results = chain.invoke({'feedback' : 'what a wonderful product really nice but not good as the other competitors'})
print(results)

