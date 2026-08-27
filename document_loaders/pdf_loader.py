from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r'C:\Users\fawwa\Desktop\langchainModels\Aircheck_Documentation.pdf')
docs = loader.lazy_load()
for document in docs:
    print(document.metadata)