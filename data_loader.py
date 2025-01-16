import os
import uuid

import streamlit as st
import vertexai
from dotenv import load_dotenv
from langchain_text_splitters import NLTKTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")  # Hosted Qdrant URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Qdrant Client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize Vertex AI for embedding
PROJECT_ID = "gen-lang-client-0404349304"
vertexai.init(project=PROJECT_ID, location="us-central1")

def embed_text(
    texts: list[str],
    task: str = "QUESTION_ANSWERING",
    model_name: str = "text-embedding-005",
    dimensionality: int | None = 768,
) -> list[list[float]]:
    """Embeds texts with a pre-trained model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]

# Create Streamlit app
st.title("Text to Qdrant Uploader")
st.write("Input text below to split it into chunks, embed it, and upload to the Qdrant cluster.")

# Input text from user
user_input = st.text_area("Enter your text here:", height=200)

if st.button("Upload to Qdrant"):
    if not user_input.strip():
        st.error("Please enter some text before uploading.")
    else:
        try:
            # Split text into chunks using NLTKTextSplitter
            text_splitter = NLTKTextSplitter(chunk_size=200)
            texts = text_splitter.split_text(user_input)
            
            # Embed and upload each chunk to Qdrant
            for row in texts:
                embedded_vector = embed_text([row])[0]
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedded_vector,
                    payload={"text": row}
                )
                qdrant_client.upsert(collection_name="Portfolio_Store", wait=True, points=[point])
            
            st.success(f"Uploaded {len(texts)} chunks to the Qdrant cluster!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Create collection if it doesn't already exist (optional)
if "Portfolio_Store" not in [col.name for col in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name="Portfolio_Store",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
