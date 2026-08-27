from langchain_community.document_loaders import WebBaseLoader
url = "https://reworn-website.vercel.app/"
loder = WebBaseLoader(url)

docs = loder.load()
print(docs[0].page_content)