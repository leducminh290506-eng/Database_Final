"""Quick script to view sample document text to understand article structure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_documents

docs = load_documents(limit=2)
for doc in docs:
    print(f"=== DOC {doc.doc_id} ===")
    print(f"Title: {doc.title}")
    print(f"Text (first 5000 chars):")
    print(doc.text[:5000])
    print()
