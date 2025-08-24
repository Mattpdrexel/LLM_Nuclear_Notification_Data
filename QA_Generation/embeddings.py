import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NuclearNotificationEmbedder:
    """
    Generates embeddings for nuclear notification data from Excel files.
    Designed for RAG (Retrieval-Augmented Generation) applications.

    Notes:
    - Each Excel row becomes a SINGLE chunk so we can reference discrete notifications.
    - Training text (e.g., cw_training) is NOT included here; keep it in a separate index/file.
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-m3",
                 cache_dir: str = "embeddings_cache"):
        """
        Initialize the embedder with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for embeddings
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # Hard cap is dictated by the underlying model (MiniLM/BERT ~512 tokens)
        try:
            self.model.max_seq_length = max(getattr(self.model, 'max_seq_length', 256), 512)
        except Exception:
            pass
        # Tokenizer for windowing
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            self.tokenizer = None
        # Windowing params tuned for long-context models like bge-m3
        # Dynamically choose a large window while staying under model/tokenizer limits
        desired_window = 4096
        tokenizer_cap = None
        try:
            tokenizer_cap = int(getattr(self.tokenizer, 'model_max_length', 0)) if self.tokenizer is not None else 0
        except Exception:
            tokenizer_cap = 0
        if tokenizer_cap and tokenizer_cap > 0 and tokenizer_cap < 10_000_000_000:
            self.max_tokens_per_window = max(1024, min(desired_window, tokenizer_cap - 32))
        else:
            self.max_tokens_per_window = 4096
        self.window_overlap = min( max(128, self.max_tokens_per_window // 8), 512 )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")
        
    def load_excel_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess Excel data.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame with cleaned data
        """
        logger.info(f"Loading Excel file: {file_path}")
        
        # Load the Excel file
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Basic cleaning
        df = df.fillna("")  # Replace NaN with empty strings
        
        # Convert all columns to string for consistent processing
        for col in df.columns:
            df[col] = df[col].astype(str)
            
        return df

    def _choose_first_present(self, row: pd.Series, candidates: List[str]) -> str:
        for c in candidates:
            for col in row.index:
                if col.lower().strip() == c.lower():
                    val = str(row[col]).strip()
                    if val and val.lower() != "nan":
                        return val
        return ""

    def _build_reference_label(self, idx: int, row: pd.Series) -> str:
        # Heuristics to construct a human-friendly reference label
        notif_id = self._choose_first_present(row, [
            "Notification", "Not.", "Notif", "NotificationID", "ID"
        ])
        short = self._choose_first_present(row, [
            "ShortText", "Short Text", "Description", "Title"
        ])
        date = self._choose_first_present(row, [
            "Date", "CreatedOn", "CreationDate", "Created", "ReportedOn"
        ])
        parts: List[str] = []
        if notif_id:
            parts.append(notif_id)
        if short:
            parts.append(short)
        label = " – ".join(parts) if parts else f"Row {idx}"
        if date:
            label = f"{label} ({date})"
        return label
    
    def create_text_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create exactly ONE chunk per DataFrame row so the full notification is captured.
        
        Args:
            df: DataFrame with notification data
            
        Returns:
            List of dictionaries with chunk data and metadata
        """
        chunks: List[Dict[str, Any]] = []
        
        for idx, row in df.iterrows():
            # Combine all columns into a single full-text notification
            text_parts: List[str] = []
            for col in df.columns:
                val = row[col]
                if val and val != "nan":
                    text_parts.append(f"{col}: {val}")
            full_text = " | ".join(text_parts)

            # Reference-friendly label for later citation
            ref_label = self._build_reference_label(idx, row)

            chunks.append({
                "id": f"row_{idx}",
                "text": full_text,                 # full notification text (no splitting)
                "row_index": int(idx),             # stable numeric index
                "columns": list(df.columns),
                "chunk_index": 0,                  # always 0 since single-chunk
                "reference_label": ref_label,      # e.g., "20926915 – ShortText (YYYY-MM-DD)"
                "source": "notifications_excel"
            })
        
        logger.info(f"Created {len(chunks)} full-notification chunks from {len(df)} rows")
        return chunks
    
    def _text_to_token_windows(self, text: str) -> List[str]:
        """Split a long text into token windows within model's context.
        Falls back to naive char-splitting if tokenizer is unavailable.
        """
        if not text:
            return [""]
        if self.tokenizer is None:
            est_token_len = max(1, len(text) // 4)
            if est_token_len <= self.max_tokens_per_window:
                return [text]
            char_window = self.max_tokens_per_window * 4
            char_stride = max(1, (self.max_tokens_per_window - self.window_overlap) * 4)
            windows: List[str] = []
            start = 0
            while start < len(text):
                end = min(len(text), start + char_window)
                windows.append(text[start:end])
                if end == len(text):
                    break
                start = start + char_stride
            return windows
        # Proper token-based split:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_tokens_per_window:
            return [text]
        windows: List[str] = []
        stride = max(1, self.max_tokens_per_window - self.window_overlap)
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self.max_tokens_per_window)
            chunk_ids = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            windows.append(chunk_text)
            if end == len(tokens):
                break
            start = start + stride
        return windows

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for text chunks.
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        # Encode each full notification once (no windowing). The underlying model
        # may truncate beyond its context length, but we do not split or chunk.
        texts: List[str] = [item.get("text", "") for item in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False,
            device=self.device,
            normalize_embeddings=True,
        )
        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        # Create result structure
        result = {
            "embeddings": embeddings_np,
            "chunks": chunks,
            "model_name": self.model_name,
            "embedding_dim": embeddings_np.shape[1],
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                "total_chunks": len(chunks),
                "device_used": str(self.device),
                "chunking": "one-row-per-chunk (no windowing; model may truncate to context)",
                "source": "notifications_only",
            }
        }
        logger.info(f"Generated embeddings with shape: {embeddings_np.shape}")
        return result
    
    def save_embeddings(self, embeddings_data: Dict[str, Any], filename: str) -> str:
        """
        Save embeddings to disk.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata
            filename: Name of the file to save
            
        Returns:
            Path to saved file
        """
        file_path = self.cache_dir / filename
        
        # Save as pickle for numpy arrays
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        # Also save metadata as JSON for easy inspection
        metadata_path = self.cache_dir / f"{filename}_metadata.json"
        metadata = {
            "model_name": embeddings_data["model_name"],
            "embedding_dim": embeddings_data["embedding_dim"],
            "generated_at": embeddings_data["generated_at"],
            "metadata": embeddings_data["metadata"],
            "chunk_count": len(embeddings_data["chunks"])
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved embeddings to: {file_path}")
        logger.info(f"Saved metadata to: {metadata_path}")
        
        return str(file_path)
    
    def load_embeddings(self, filename: str) -> Dict[str, Any]:
        """
        Load embeddings from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        file_path = self.cache_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        logger.info(f"Loaded embeddings from: {file_path}")
        return embeddings_data
    
    def process_excel_file(self, 
                          excel_path: str, 
                          output_filename: str = "nuclear_notifications_embeddings.pkl") -> str:
        """
        Complete pipeline to process Excel file and generate embeddings.
        
        Args:
            excel_path: Path to Excel file
            output_filename: Name for output embeddings file
            
        Returns:
            Path to saved embeddings file
        """
        logger.info("Starting embedding generation pipeline")
        
        # Load data
        df = self.load_excel_data(excel_path)
        
        # Create one chunk per row (full notification)
        chunks = self.create_text_chunks(df)
        
        # Generate embeddings
        embeddings_data = self.generate_embeddings(chunks)
        
        # Save embeddings
        output_path = self.save_embeddings(embeddings_data, output_filename)
        
        logger.info("Embedding generation pipeline completed")
        return output_path


def main():
    """Main function to run the embedding generation."""
    
    # Configuration
    excel_file = "raw_data/salem_cw_data.xlsx"
    output_file = "nuclear_notifications_embeddings.pkl"
    
    # Initialize embedder
    embedder = NuclearNotificationEmbedder(
        model_name="BAAI/bge-m3",  # Good balance of speed/quality
        cache_dir="embeddings_cache"
    )
    
    # Process the Excel file
    try:
        output_path = embedder.process_excel_file(
            excel_path=excel_file,
            output_filename=output_file
        )
        print(f"✅ Embeddings generated successfully: {output_path}")
        
        # Load and display some statistics
        embeddings_data = embedder.load_embeddings(output_file)
        print(f"📊 Statistics:")
        print(f"   - Total chunks: {len(embeddings_data['chunks'])}")
        print(f"   - Embedding dimension: {embeddings_data['embedding_dim']}")
        print(f"   - Model used: {embeddings_data['model_name']}")
        
        # Show a sample chunk
        if embeddings_data['chunks']:
            sample = embeddings_data['chunks'][0]
            print(f"📝 Sample reference: {sample.get('reference_label', 'N/A')}")
            print(f"📝 Sample text (first 200 chars):")
            print(f"   {sample['text'][:200]}...")
            
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        raise


if __name__ == "__main__":
    main()
