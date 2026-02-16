
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGODB_URL, MONGODB_DATABASE

async def setup_files_collection():
    print(f"Connecting to MongoDB at {MONGODB_URL}...")
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DATABASE]
    
    collection_name = "files"
    if collection_name not in await db.list_collection_names():
        print(f"Creating collection '{collection_name}'...")
        await db.create_collection(collection_name)
    else:
        print(f"Collection '{collection_name}' already exists.")
    
    files_collection = db[collection_name]
    
    # Create indexes
    print("Creating indexes...")
    await files_collection.create_index([("fileId", 1)], unique=True)
    await files_collection.create_index([("createdBy", 1)])
    await files_collection.create_index([("isVectorized", 1)])
    await files_collection.create_index([("isDeleted", 1)])
    
    # Compound index for queries
    await files_collection.create_index([
        ("createdBy", 1), 
        ("isVectorized", 1), 
        ("isDeleted", 1)
    ])
    
    print("Indexes created successfully.")
    
    # Verify
    indexes = await files_collection.list_indexes().to_list(length=None)
    for index in indexes:
        print(f" - {index['name']}: {index['key']}")

if __name__ == "__main__":
    asyncio.run(setup_files_collection())
