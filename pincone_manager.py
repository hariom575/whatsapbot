import pinecone
from sentence_transformers import SentenceTransformer
import PyPDF2

# import website_crawler

class PineconeClient:
    def __init__(self):
        pinecone.init(api_key="1f610917-a629-4f62-b66f-eac22fd09a81",
                      environment="us-east4-gcp")
        
    def insert_into_index(self, content,pageNo):
        print(content)
        model = SentenceTransformer("average_word_embeddings_glove.6B.300d")
        # pinecone.create_index("pinconetutorial",dimension=300)
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
    
if __name__ == '__main__':
    file = open('../Downloads/TheTeachingofBuddha.pdf', 'rb')
    reader = PyPDF2.PdfReader(file)
    page1 = reader.pages[1]
    for i in range (0,len(reader.pages)):
        pineconeClient = PineconeClient()
        pineconeClient.insert_into_index(reader.pages[i].extract_text(),i)
