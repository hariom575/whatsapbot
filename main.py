# Third-party imports
import openai
from fastapi import FastAPI, Form, Depends, Request
from decouple import config
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
import pinecone
from sentence_transformers import SentenceTransformer
import PyPDF2

# Internal imports
from models import Conversation, SessionLocal
from utils import send_message, logger

app = FastAPI()
# Set up the OpenAI API client
openai.api_key = config("OPENAI_API_KEY")

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information, "
    "answer the question: {query_str}. "
    "Keep the length of the answer to 4-6 sentences.\n"
    "Do not answer questions about the following topics: \n"
    "porn, politics, religion, violence, racism, sexism, etc.\n"
    "If you cannot answer the question, answer with the following response: \n"
    "Sorry, I cannot answer this question. Please try a different conversation. \n"
    "If the question is about a code snippet, please provide the code snippet as well.\n"
)

class PineconeClient:
    def __init__(self):
        pinecone.init(api_key="1f610917-a629-4f62-b66f-eac22fd09a81",
                      environment="us-east4-gcp")
        
    def insert_into_index(self, content,pageNo):
        print(content)
        model = SentenceTransformer("average_word_embeddings_glove.6B.300d")
        # pinecone.create_index("pincone-test",dimension=300)
        index = pinecone.Index(index_name="pinconetutorial")
        print(index)
        chunk_size = 4000
        for i in range(0,len(content),chunk_size):
            vector = (lambda x: model.encode(
                str(x)).tolist())(content[i:i+chunk_size])
            print(content[i:i+chunk_size])
            print(str(pageNo))
            response = index.upsert(vectors=[
                {
                    'id': str(pageNo),
                    'values': vector,
                    'metadata': {
                        "content": content[i:i+chunk_size]
                    }
                }
            ])
            print(response)

    def query_from_index(self, question, index_name = "pinconetutorial"):
        query_questions = []
        query_questions.append(question)
        index = pinecone.Index(index_name=index_name)
        model = SentenceTransformer("average_word_embeddings_glove.6B.300d")
        query_vectors = [model.encode(str(question)).tolist()
                         for question in query_questions]

    
        query_results = index.query(
                vector=query_vectors,
                top_k=5,
                include_metadata=True
                )
        if len(query_results["matches"])==0 or query_results["matches"][0]["score"] < 0.1:
            return None
        
        print(query_results)
        return query_results["matches"][0]["metadata"]["content"] , query_results["matches"][0]["id"]


# Dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.get("/")
async def index():
    return {"msg": "working"}

@app.post("/message")
async def reply(request: Request, Body: str = Form(), db: Session = Depends(get_db)):
    # Extract the phone number from the incoming webhook request
    form_data = await request.form()
    whatsapp_number = form_data['From'].split("whatsapp:")[-1]
    print(f"Sending the ChatGPT response to this number: {whatsapp_number}")
    print(Body)

    # Call the OpenAI API to generate text with ChatGPT
    pineconeClient = PineconeClient()
    context_data, pageNo = pineconeClient.query_from_index(Body)
    print(context_data)
    prompt = DEFAULT_TEXT_QA_PROMPT_TMPL.format(context_str=context_data, query_str = Body)
    print(prompt)

    response = openai.Completion.create(
            engine = 'text-davinci-003',
            prompt = prompt,
            max_tokens = 400,
            temperature = 0.5)

    # The generated text
    chatgpt_response = response["choices"][0]["text"]


    # Store the conversation in the database
    try:
        conversation = Conversation(
            sender=whatsapp_number,
            message=Body,
            response=chatgpt_response
            )
        db.add(conversation)
        db.commit()
        logger.info(f"Conversation #{conversation.id} stored in database")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error storing conversation in database: {e}")
    send_message(whatsapp_number, chatgpt_response)
    return ""


