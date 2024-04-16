#make a restaurant menu rag and then call ollama
#Steps
#1.Create a JSON menu using maybe GPT4 or Haiku
#2. Create a vector index for this json

from flask import Flask, render_template, request, jsonify
import json
import os
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage, Document, get_response_synthesizer, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import ollama
from ollama import Client


def create_document_from_json_item(json_item):
    ti = json_item['title'] 
    des = json_item['description']
    keys = json_item['keywords']


    # if des:  # depending on the menu/embedding model, this could be helpful 
    #     ti += des


    if keys: 
        ti += "keywords:" + keys
    document = Document(text=ti, metadata=json_item)
    return document


def generate_embeddings_for_document(document, model_name="BAAI/bge-small-en-v1.5"):
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    embeddings = embed_model.get_text_embedding(document.text)
    return embeddings


file_path = "./menu.json"
index = None 


if not os.path.exists("./index"):        
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)


    documents = []
    for item in json_data:
        document = create_document_from_json_item(item)
        document_embeddings = generate_embeddings_for_document(document)
        document.embedding = document_embeddings
        documents.append(document)


    Settings.llm =Ollama(model="llama2", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="./index")
else:
    
    Settings.llm = Ollama(model="llama2", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    storage_context = StorageContext.from_defaults(persist_dir="./index")
    
    index = load_index_from_storage(storage_context)


# Now querying the vector index
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
# service_context = ServiceContext.from_defaults(llm=None, embed_model='local')
response_synthesizer = get_response_synthesizer()
query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)])


def parse_response_to_json(response_str):
    items = response_str.split("title: ")[1:]  # Split the response and ignore the first empty chunk
    json_list = []


    for item in items:
        lines = item.strip().split('\n')
        item_json = {
            "title": lines[0].strip(),
            "description": lines[1].replace("description: ", "").strip(),
            "keywords": lines[2].replace("keywords: ", "").strip(),
            "page": int(lines[3].replace("page: ", "").strip())
        }
        json_list.append(item_json)


    return json_list


def query_index(query):
    print("reached the def for query engine")
    dummy_response  = query_engine.query("tell em what on menu today?")
    print("i am dummy response",dummy_response)
    response = query_engine.query(query)
    # Original results parsed to JSON
    print("i am json fo the query", response)
    return parse_response_to_json(str(response))


#Creating a flask app for server

app = Flask(__name__)


@app.route('/chat')
def index():
    return render_template('chat.html')


@app.route('/')
def menu():
    return "hey there"


def describe_items(json_list):
    description_str = "Some possible items you might be interested in include the following:<br><br>"
    for item in json_list:
        description_str += f"<strong>{item['title']}</strong> - {item['description']}<br><br>"
    return description_str


def generate_response_llm(query, original_res):
    # Generating prompt for GPT
    prompt = f"This is a user at a restaurant searching for items to order. Given these initial results {original_res} for the following user query '{query}', return the JSON object for the items that make sense to include as a response (e.g., remove only items that are not at all relevant to the query='{query}') -- keep in mind that they may all be relevant and its perfectly fine to not remove any items. YOU MUST RETURN THE RESULT IN JSON FORM"
    print("i am sending query to ollama")
    client = Client(host='http://localhost:11434')
    response = client.chat(
        model="llama2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    print("i got the response from ollama", response)
    
    if response.choices:
        reply = response.choices[0].message.content
        filtered_res_json_str = re.search(r"```json(.+?)```", reply, re.DOTALL)


        print(filtered_res_json_str)
        if filtered_res_json_str:
            filtered_res_json = json.loads(filtered_res_json_str.group(1))
            if not len(filtered_res_json): 
                return original_res
        else:
            filtered_res_json = original_res
        
        
        return filtered_res_json
    else:
        return original_res




@app.route('/search', methods=['POST'])
def search():
    print("i reached search")
    query = request.form.get('query')
    print("i am query", query)
    original_res = query_index(query)
    filtered_res_json = generate_response_llm(query, original_res)
    # st.log({"query": query, "results": describe_items(filtered_res_json)})
    return jsonify({'res': describe_items(filtered_res_json)})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
