# src/phase1/semantic_chunker.py
"""
Semantic chunker for guides.
Reads `data/guides/guides.json` (the format produced by your scraper),
splits guide sections into semantic chunks ~150-350 tokens (approx by words),
applies overlap, and writes `data/processed/guides_chunks.json`.

Outputs a list of chunk objects with metadata:
{
  "chunk_id": "GUIDA_05__section_2__chunk_0",
  "guide_number": "GUIDA 05",
  "guide_title": "I VETRI",
  "section_index": 1,
  "section_heading": "Conoscere lo Sporco dei Vetri",
  "chunk_index": 0,
  "chunk_text": "...",
  "products_mentioned": [...],
  "url": "...",
  "word_count": 210
}
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import re
import uuid
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import GUIDES_COMBINED_FILE, PROCESSED_DIR, CHUNK_TARGET_WORDS, CHUNK_OVERLAP_WORDS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_whitespace(text: str) -> str:
    text = text or ""
    # replace CRLF and multiple spaces
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def split_paragraphs(section_content: str) -> List[str]:
    """
    Split a section into paragraphs. Preserve paragraphs that appear meaningful.
    """
    if not section_content:
        return []
    # split on double newlines first, then fallback to single newline
    paras = [p.strip() for p in re.split(r'\n{2,}', section_content) if p.strip()]
    if not paras:
        paras = [p.strip() for p in re.split(r'\n', section_content) if p.strip()]
    return paras


def chunk_paragraph(paragraph: str, target_words: int, overlap_words: int) -> List[str]:
    """
    Chunk a single paragraph into chunks of roughly target_words
    with overlap of overlap_words.
    """
    words = paragraph.split()
    n = len(words)
    if n == 0:
        return []
    if n <= target_words:
        return [" ".join(words)]
    chunks = []
    start = 0
    while start < n:
        end = min(start + target_words, n)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == n:
            break
        # advance by target - overlap
        start = max(0, end - overlap_words)
    return chunks


def chunk_section(section_text: str, target_words: int, overlap_words: int) -> List[str]:
    """
    Chunk a section by paragraphs, then chunk paragraphs further.
    """
    paras = split_paragraphs(section_text)
    if not paras:
        return []
    all_chunks = []
    for para in paras:
        para = normalize_whitespace(para)
        # if paragraph is small, keep as-is
        if len(para.split()) <= target_words:
            all_chunks.append(para)
            continue
        # else split paragraph into smaller chunks
        parag_chunks = chunk_paragraph(para, target_words, overlap_words)
        all_chunks.extend(parag_chunks)
    return all_chunks


def build_chunks_from_guides(guides: List[Dict[str, Any]],
                             target_words: int = CHUNK_TARGET_WORDS,
                             overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[Dict[str, Any]]:
    """
    Build a list of chunk dicts from the guides JSON structure.
    """
    chunks = []
    for guide in guides:
        guide_number = guide.get("guide_number") or guide.get("id") or ""
        guide_title = guide.get("title") or ""
        guide_url = guide.get("url", "")
        sections = guide.get("sections", []) or []
        for s_idx, section in enumerate(sections):
            section_heading = section.get("heading", "") or ""
            section_text = section.get("content", "") or ""
            products = section.get("products_mentioned", []) or []
            # if section is empty but there's intro/description, fallback
            if not section_text:
                # try to use guide-level intro or description
                section_text = section.get("content", "") or guide.get("intro", "") or guide.get("description", "") or ""
            # normalize
            section_text = normalize_whitespace(section_text)
            # skip very short
            if len(section_text.strip()) < 10:
                continue
            section_chunks = chunk_section(section_text, target_words, overlap_words)
            for c_idx, chunk_text in enumerate(section_chunks):
                chunk_id = f"{guide_number}__sec{s_idx}__chunk{c_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "guide_number": guide_number,
                    "guide_title": guide_title,
                    "section_index": s_idx,
                    "section_heading": section_heading,
                    "chunk_index": c_idx,
                    "chunk_text": chunk_text,
                    "products_mentioned": products,
                    "url": guide_url,
                    "word_count": len(chunk_text.split())
                })
    return chunks


def load_guides(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Guides file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        guides = json.load(f)
    return guides


def save_chunks(chunks: List[Dict[str, Any]], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    guides_file = Path(GUIDES_COMBINED_FILE)
    out_file = Path(PROCESSED_DIR) / "guides_chunks.json"

    logger.info(f"Loading guides from: {guides_file}")
    guides = load_guides(guides_file)
    logger.info(f"Loaded {len(guides)} guides")

    logger.info("Building chunks...")
    chunks = build_chunks_from_guides(guides)
    logger.info(f"Built {len(chunks)} chunks")

    logger.info(f"Saving chunks to: {out_file}")
    save_chunks(chunks, out_file)
    logger.info("Done.")


if __name__ == "__main__":
    main()
