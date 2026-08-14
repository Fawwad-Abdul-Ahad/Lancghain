from typing import Literal, Optional, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# dynamic prompts
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

class Review (TypedDict):
    key_themes: Annotated[list[str],"Write down all the key themes in discussed in the reiew"] 
    summary: Annotated[str,'Write a comprehensive summary of the given review']
    sentiment: Annotated[Literal['pos','neg'], 'Write a sentiment about the given review']
    pros : Annotated[Optional[str], 'Write the pros of the given review']
    cons : Annotated[Optional[str], 'Write the cons of te given review']


structured_model = model.with_structured_output(Review)

response = structured_model.invoke("I recently purchased these wireless earbuds and I am extremely impressed with the overall quality. The sound is clear, the bass is powerful, and the battery lasts much longer than I expected. I have been using them daily for work calls, music, and workouts without any issues. The charging case is compact and easy to carry. For the price, this is one of the best purchases I have made this year.")
print(response['cons'])