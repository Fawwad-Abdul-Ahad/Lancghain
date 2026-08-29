from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


text = """
**from langchain_text_splitters import RecursiveCharacterTextSplitter

#text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=100,
    chunk_overlap=0
)

/n/nresult = text_splitter.split_text(text)

print(result)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_overlap = 0,
    chunk_size = 50,
)

result = splitter.split_text(text)
print(len(result))
print(result)
