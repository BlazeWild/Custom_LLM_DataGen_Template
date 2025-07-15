from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore
import json
import glob
import os

def main():
    """
    Process all PDFs and save chunks to chunk_folder for later processing
    """
    # Create chunk folder if it doesn't exist
    os.makedirs("chunks", exist_ok=True)
    
    converter = DocumentConverter()
    
    # Get all PDF files in the data directory
    pdf_files = glob.glob("data/*.pdf")
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    
    if len(pdf_files) == 0:
        print("No PDF files found in data/ directory. Exiting.")
        return
    
    # Configure chunker for more, smaller chunks
    chunker = HybridChunker(
        chunk_size=2000,      # Smaller chunks
        chunk_overlap=50,    # Some overlap to maintain context
    )
    
    # Process each PDF file
    all_chunks = []
    chunk_metadata = []
    
    for pdf_file in pdf_files:
        print(f"{Fore.CYAN}Processing: {pdf_file}{Fore.RESET}")
        try:
            doc = converter.convert(pdf_file).document
            chunks = list(chunker.chunk(dl_doc=doc))
            
            for chunk_idx, chunk in enumerate(chunks):
                # Save individual chunk in JSON format
                chunk_filename = f"{os.path.splitext(os.path.basename(pdf_file))[0]}_chunk_{chunk_idx:03d}.json"
                chunk_path = os.path.join("chunks", chunk_filename)
                
                # Contextualize the chunk
                print(f"  -> Contextualizing chunk {chunk_idx + 1}/{len(chunks)}")
                enriched_text = chunker.contextualize(chunk=chunk)
                
                # Create JSON structure like the example
                chunk_data = {
                    "source_file": pdf_file,
                    "chunk_index": chunk_idx,
                    "raw_text": chunk.text,
                    "contextualized_text": enriched_text,
                    "metadata": {
                        "chunk_size": len(chunk.text),
                        "contextualized_size": len(enriched_text)
                    }
                }
                
                # Save chunk to JSON file
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, indent=2, ensure_ascii=False)
                
                # Store metadata
                chunk_metadata.append({
                    "chunk_id": len(all_chunks),
                    "source_pdf": pdf_file,
                    "chunk_filename": chunk_filename,
                    "raw_text_preview": chunk.text[:100] + "...",
                    "contextualized_preview": enriched_text[:100] + "..."
                })
                
                all_chunks.append(enriched_text)
            
            print(f"  -> {Fore.GREEN}Added {len(chunks)} chunks from {pdf_file}{Fore.RESET}")
            
        except Exception as e:
            print(f"{Fore.RED}Error processing {pdf_file}: {e}{Fore.RESET}")
            continue
    
    # Save metadata
    metadata_path = os.path.join("chunks", "chunks_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(all_chunks),
            "source_pdfs": pdf_files,
            "chunks": chunk_metadata
        }, f, indent=2)
    
    print(f"\n{Fore.GREEN}✓ Chunking complete!{Fore.RESET}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Chunks saved to: chunks/")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nNext step: Run syntheticdatageneration.py to generate Q&A pairs from chunks")

if __name__ == "__main__":
    main()