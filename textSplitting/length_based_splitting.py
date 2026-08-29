from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
    Skip to main content
scikit-learn homepage
Install
User Guide
API
Examples
Community
Examples
SVM: Separating hyperplane for unbalanced classes
SVM: Weighted samples
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=0,
    separators=''
)

result = text_splitter.split_text(text)
print(result[0])