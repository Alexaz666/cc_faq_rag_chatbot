# src/build_index.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_data
from src.text_splitter import split_text
from src.vector_store import create_vector_db


def build_index(
    json_path: str,
    persist_dir: str = "./chroma_db/cc_faq_openai",
    collection_name: str = "cc_faq",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    dryrun: bool = False,
    limit: int=0
) -> None:
    # 1) Load
    scraped_data = load_data(json_path)

    # 2) Split
    chunked_data = split_text(
        scraped_data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"Created {len(chunked_data)} unique chunks.")

    if dryrun:
        print("Dry run enabled - skipping embedding")
        print("Sample chunk:")
        print(chunked_data[0])
        return
    
    if limit and limit > 0:
        chunked_data = chunked_data[:limit]
        print(f"Limiting to {limit} chunks for this run.")

    # 3) Embed + persist
    vectorstore = create_vector_db(
        chunked_data,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    # 4) Smoke check
    print(f"Persist dir: {persist_dir}")
    print(f"Collection '{collection_name}' chunk count: {vectorstore._collection.count()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default="data/scraped_data_w_segment.json",
        help="Path to scraped_data_w_segment.json",
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db/cc_faq_openai",
        help="Directory to persist Chroma DB",
    )
    parser.add_argument(
        "--collection",
        default="cc_faq",
        help="Chroma collection name",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    
    # For testing 
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="If set, skip embedding and just print a sample chunk.",
    )
    parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="Limit number of chunks to embed (0 = embed all).",
)

    args = parser.parse_args()

    # Quick sanity check the file exists
    if not Path(args.json).exists():
        raise FileNotFoundError(f"JSON not found: {args.json}") 

    build_index(
        json_path=args.json,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dryrun=args.dryrun,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
