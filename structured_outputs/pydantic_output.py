
from typing import Literal, Optional, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field 
# dynamic prompts
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

class Review (BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes in discussed in the reiew")
    summary: str = Field(description = "Write a comprehensive summary of the given review")
    sentiment : str = Field(description = 'Write a sentiment about the given review')
    pros : Optional[str] = None 
    cons : Optional[str] = None


structured_model = model.with_structured_output(Review)

response = structured_model.invoke("I recently purchased these wireless earbuds and I am extremely impressed with the overall quality. The sound is clear, the bass is powerful, and the battery lasts much longer than I expected. I have been using them daily for work calls, music, and workouts without any issues. The charging case is compact and easy to carry. For the price, this is one of the best purchases I have made this year.")
print(response)