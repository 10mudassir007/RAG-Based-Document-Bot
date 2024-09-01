from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import tempfile



custom_model_path = r"f:\Files\NOTES etc\ASSIGNMENT\models"

tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
model = AutoModelForCausalLM.from_pretrained(custom_model_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=100)

llm = HuggingFacePipeline(pipeline=pipe)

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
clear_chat = st.sidebar.button("Clear Chat")

st.title("RAG based Document Bot")
query = st.chat_input("Enter your query:")

if clear_chat:
    st.session_state.messages = []


if uploaded_file is not None:
    # Use a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load_and_split()



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embedding)

    template = """
    Context: {context}

    Question: {question}

    Answer the question based on the context above.
    """
    retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':2})

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm,prompt=prompt)

    question = "Is hate discussed in The Subtle Art of Not Giving a F*ck?"
    content = retriever.get_relevant_documents(query=question)
    context = ''.join(content[i].page_content for i in range(len(content)))


    def get_answer(query):
        response = llm_chain.run({'context': context,'question':query})
        answer_start = response.find("Answer the question based on the context above.")
        answer_part = response[answer_start + len("Answer the question based on the context above."):].strip()
        answer_part = answer_part.replace('\n', ' ').strip()
        return answer_part

    

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])
    
    if query:
        st.chat_message("user").markdown(query)
        response = get_answer(query)
        st.session_state.messages.append({'role':'user','content':query})

        st.chat_message('ai').markdown(response)

        st.session_state.messages.append({'role':'ai','content':response})
else:
    st.sidebar.info("Please upload a PDF document to start chatting.")