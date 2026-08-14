from typing import TypedDict

class typing (TypedDict):
    name:str
    age:int


new_typing = typing(name="hello", age=12)
print(new_typing)