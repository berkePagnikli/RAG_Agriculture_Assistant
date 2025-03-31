from qdrant_client import QdrantClient
from autogen import AssistantAgent, config_list_from_json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Initialize components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host="localhost", port=6333)
config_list = config_list_from_json("OAI_CONFIG_LIST")

# Retrieve data from vector database (Qdrant in this case)
def retrieve_urban_data(query: str, collection_names: List[str], top_k: int = 3) -> List[Dict]:
    query_embedding = embedding_model.encode(query).tolist()
    
    results = []
    for collection in collection_names:
        hits = client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        for hit in hits:
            results.append({
                "content": hit.payload["text_description"],
                "metadata": {
                    "type": hit.payload["data_type"],
                    "id": hit.payload["id"],
                    "score": hit.score
                }
            })
    
    return sorted(results, key=lambda x: x["metadata"]["score"], reverse=True)[:top_k*2]

# Create the RAG agent
# This agent will use the retrieved context to answer questions
assistant = AssistantAgent(
    name="assistant",
    system_message="""
    # Overview
        - You are an expert in urban agriculture. Use the provided context to answer questions.
    
    # Guideline
        - You will recieve user query in the format: "User: {query}" and context in the format: "Context: {context}". Context is a list of documents with their metadata. Treat context as if you retrieved it.
        - If multiple methods exist, compare them using their metrics.
        - Mention limitations from disadvantages when relevant.
        - Prioritize quantitative data (space efficiency, costs).
        - Do not output the retrieved data directly. Analyze it and present a comprehensive explanation and a summary.
        - If unsure, request clarification about specific urban context.
    """,
    llm_config={
        "config_list": config_list,
        "timeout": 600,
    }
)

# Create a function to run the agent with user input and retrieved context
def runAgent(question: str):
    collections = ["urban_ag_methods", "urban_crops", "urban_challenges", "urban_best_practices"]
    context_data = retrieve_urban_data(question, collections)
    context_str = "\n".join([
        f"Document {i+1} ({doc['metadata']['type']}, score: {doc['metadata']['score']:.2f}):\n{doc['content']}"
        for i, doc in enumerate(context_data)
    ])
    print("Context retrieved:")
    print(context_str)
    agent_input = (f"User:\n{question}\n\n"
                   f"Context:\n{context_str}")
    messages = [
        {"role": "user", "content": agent_input}
    ]

    response = assistant.generate_reply(messages)

    print("Agent Response:")
    print(response)

runAgent("What's the most space-efficient method for growing strawberries in a polluted urban area?")