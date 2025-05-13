from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
from dotenv import load_dotenv
load_dotenv()

# load documents
def load_dir(path):
    '''function to load the documents from the directory'''
    loader = PyPDFLoader(path)
    document = loader.load()
    return document

# passing the directory path to the function
document = load_dir("D:/machine_learning/MCQ creator app/doc/document.pdf")


def split_docs(document, chunk_size = 1000, chunk_overlap = 20):
    '''function to split the document into chunks'''
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(document)
    return docs
docs = split_docs(document)


# generate text embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vector store --> pinecone
index = PineconeVectorStore.from_documents(
    docs,
    embeddings,
    index_name="mcq-creator",
)
def get_similarity_search(query, k=3):
    '''function to get the most similar documents to the query'''
    results = index.similarity_search(query, k=k)
    return results


# load the chain
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation",
)

retriever = index.as_retriever()
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# this function will help us get the answer to the question that we raise
def get_answer(query):
    '''function to get the answer to the question'''
    # doc = get_similarity_search(query)
    answer = chain.invoke({
    # "input_documents": doc,
    "query": query
    })
    return answer

our_query = "what kind of token is nft??"
answer = get_answer(our_query)
# print(answer)

# get structured output in MCQ format
response_schemas = [
    ResponseSchema(
        name="question", description="Question generated from provided input text data."
    ),
    ResponseSchema(
        name="choices",
        description="Available options for a multiple-choice question in comma separated.",
    ),
    ResponseSchema(name="answer", description="Correct answer for the asked question."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# giving prompt to the model
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(
            """When a text input is given by the user, please generate multiple choice questions 
        from it along with the correct answer. 
        \n{format_instructions}\n{user_prompt}"""
        )
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions},
)
final_query = prompt.format_prompt(user_prompt=answer["result"])
query_output = llm.invoke(final_query.to_messages())

# removing the extra characters from the output
json_string = re.search(r"{(.*?)}", query_output, re.DOTALL).group(1)
print(json_string)


index.delete(delete_all=True)
