from flask import Flask, render_template, request, jsonify



import os
os.environ["OPENAI_API_KEY"] ="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"
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader ,UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

# Retrieve Data
def get_docs():
    # loader = WebBaseLoader('https://www.w3schools.com/gen_ai/chatgpt-3-5/index.php')
    # docs = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=200,
    #     chunk_overlap=20
    # )

    # splitDocs = text_splitter.split_documents(docs)
    # return splitDocs
    
    #pdf
    pdf_loader = UnstructuredPDFLoader("Chapter06MobileCommerceandUbiquitousComputing.pdf")
    pdf_pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(pdf_pages)
    return texts

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )

    prompt = ChatPromptTemplate.from_template("""Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng.
        Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
        Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt

        Context: {context}
        Question: {input}

    """)

    # chain = prompt | model
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 2})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


docs = get_docs()
# print(docs)
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

# response = chain.invoke({
#     "input": "phát hiện ảnh giả mạo trong bài báo như thế nào?",
# })

# print(response["answer"])
def get_response(chain, input_text):
    # Hàm này sẽ gọi chain.invoke() với input_text được truyền vào và trả về response
    response = chain.invoke({"input": input_text})
    return response["answer"]



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])

def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_respone(input)

def get_Chat_respone(text):

    # Let's chat for 5 lines
    for step in range(5):
        question = str(text)
        if not question:
            break
        response = get_response(chain, question)
        return response

if __name__ == '__main__':
    docs = get_docs()
    vectorStore = create_vector_store(docs)
    chain = create_chain(vectorStore)
    app.run(debug=True)