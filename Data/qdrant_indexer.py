from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from Data.data import (urban_ag_data,urban_crop_data,urban_challenges,urban_best_practices)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Create collections for each data type
def create_collection(collection_name: str, vector_size: int = 384) -> None:
    """Create a collection in Qdrant if it doesn't exist"""
    try:
        # Check if collection exists
        client.get_collection(collection_name=collection_name)
        print(f"Collection {collection_name} already exists")
    except Exception:
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection {collection_name}")

# Create collections for each data type
create_collection("urban_ag_methods")
create_collection("urban_crops")
create_collection("urban_challenges")
create_collection("urban_best_practices")

# Function to prepare data for indexing
def create_document_text(data: Dict[str, Any], data_type: str) -> str:
    """Convert a dictionary to a descriptive text for embedding"""
    if data_type == "urban_ag_methods":
        text = f"{data['description']} "
        text += f"Suitable crops: {', '.join(data['suitable_crops'])}. "
        text += f"Space efficiency: {data['space_efficiency']}, "
        text += f"Water usage: {data['water_usage']}, "
        text += f"Energy requirement: {data['energy_requirement']}. "
        text += f"Advantages: {', '.join(data['advantages'])}. "
        text += f"Disadvantages: {', '.join(data['disadvantages'])}. "
        text += f"Ideal locations: {', '.join(data['ideal_locations'])}. "
        text += f"Startup cost range: {data['startup_cost_range']}."
    
    elif data_type == "urban_crops":
        text = f"Crop with growth cycle of {data['growth_cycle_days']} days. "
        text += f"Yield per sqft: {data['yield_per_sqft']}, "
        text += f"Water needs: {data['water_needs']}, "
        text += f"Light needs: {data['light_needs']}, "
        text += f"Temperature range: {data['temperature_range'][0]} to {data['temperature_range'][1]} degrees. "
        text += f"Difficulty: {data['difficulty']}, Market value: {data['market_value']}. "
        text += f"Nutritional highlights: {data['nutritional_highlights']}. "
        text += f"Best growing methods: {', '.join(data['best_growing_methods'])}. "
        text += f"Companion plants: {', '.join(data['companion_plants'])}. "
        text += f"Seasonal notes: {data['seasonal_notes']}."
    
    elif data_type == "urban_challenges":
        text = f"{data['description']}. "
        text += f"Solutions: {', '.join(data['solutions'])}. "
        text += f"Impact score: {data['impact_score']}. "
        text += f"Detailed strategies: {'. '.join(data['detailed_strategies'])}. "
        text += f"Recommended resources: {', '.join(data['recommended_resources'])}."
    
    elif data_type == "urban_best_practices":
        text = f"{data['title']}. "
        text += f"Practices: {'. '.join(data['practices'])}."
    
    return text

# Function to index data
def index_data(data_dict: Dict[str, Dict[str, Any]], collection_name: str, data_type: str) -> None:
    """Index data into a Qdrant collection"""
    # Prepare points for batch upload
    points = []
    
    for i, (key, value) in enumerate(data_dict.items()):
        # Create descriptive text for embedding
        text = create_document_text(value, data_type)
        
        # Generate embedding
        embedding = model.encode(text).tolist()
        
        # Add full data dict as payload
        payload = {
            "id": key,
            "data": value,
            "data_type": data_type,
            "text_description": text
        }
        
        # Create point
        point = models.PointStruct(
            id=i,
            vector=embedding,
            payload=payload
        )
        
        points.append(point)
    
    # Upload points in batch
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Indexed {len(points)} items in {collection_name}")

# Index all data types
index_data(urban_ag_data, "urban_ag_methods", "urban_ag_methods")
index_data(urban_crop_data, "urban_crops", "urban_crops")
index_data(urban_challenges, "urban_challenges", "urban_challenges")
index_data(urban_best_practices, "urban_best_practices", "urban_best_practices")