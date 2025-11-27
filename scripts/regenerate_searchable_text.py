"""Regenerate searchable_text for guides with deduplication."""
import json
from pathlib import Path

def create_searchable_text(guide):
    """Create clean searchable text from guide for vector embedding."""
    parts = []
    seen_content = set()
    
    def add_unique_content(text, prefix=''):
        if not text or len(text.strip()) < 20:
            return False
        key = text.strip()[:100].lower()
        if key in seen_content:
            return False
        seen_content.add(key)
        parts.append(f'{prefix}{text}' if prefix else text)
        return True
    
    if guide.get('title'):
        parts.append(f"Guida: {guide['title']}")
    if guide.get('description'):
        desc = ' '.join(guide['description'].split())
        add_unique_content(desc, 'Descrizione: ')
    
    if guide.get('intro'):
        intro = ' '.join(guide['intro'].split())
        add_unique_content(intro, '\nIntroduzione:\n')
    
    for section in guide.get('sections', []):
        heading = section.get('heading', '').strip()
        content = section.get('content', '').strip()
        
        if not content or len(content) < 50:
            continue
        if content.lower() == heading.lower():
            continue
        
        if heading and heading.lower() not in ['indice', 'index', '']:
            if not any(heading.lower() in seen.lower() for seen in seen_content if len(seen) > 20):
                parts.append(f'\n## {heading}')
        
        add_unique_content(content)
    
    if guide.get('tips'):
        parts.append('\n## Note e Suggerimenti')
        for tip in guide['tips']:
            tip_clean = tip.strip()
            if tip_clean and len(tip_clean) > 20:
                tip_key = tip_clean[:80].lower()
                if tip_key not in seen_content:
                    seen_content.add(tip_key)
                    parts.append(f'• {tip_clean}')
    
    if guide.get('products_mentioned'):
        product_names = [p['name'] for p in guide['products_mentioned'] 
                      if p.get('name') and 'ACQUISTA' not in p.get('name', '').upper()]
        if product_names:
            parts.append(f'\nProdotti consigliati: {", ".join(product_names)}')
    
    return '\n\n'.join(parts)

def main():
    # Load existing guides
    guides_file = Path('data/guides/guides.json')
    with open(guides_file, 'r', encoding='utf-8') as f:
        guides = json.load(f)
    
    print(f"Processing {len(guides)} guides...\n")
    
    # Regenerate searchable_text
    total_old = 0
    total_new = 0
    
    for guide in guides:
        old_text = guide.get('searchable_text', '')
        new_text = create_searchable_text(guide)
        guide['searchable_text'] = new_text
        
        old_len = len(old_text)
        new_len = len(new_text)
        total_old += old_len
        total_new += new_len
        
        reduction = ((old_len - new_len) / old_len * 100) if old_len > 0 else 0
        print(f"{guide['title']}: {old_len:,} -> {new_len:,} chars ({reduction:.1f}% reduction)")
    
    # Save combined file
    with open(guides_file, 'w', encoding='utf-8') as f:
        json.dump(guides, f, ensure_ascii=False, indent=2)
    
    # Also save individual files
    individual_dir = Path('data/guides/individual')
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    import re
    for guide in guides:
        # Clean title for filename (remove newlines, special chars)
        title_clean = re.sub(r'[\n\r\s]+', '_', guide['title']).strip('_')
        title_slug = re.sub(r'[^\w\-]', '', title_clean.lower())
        guide_num = guide['guide_number'].replace(' ', '_')
        filename = f"{guide_num}_{title_slug}.json"
        filepath = individual_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(guide, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Total: {total_old:,} -> {total_new:,} chars")
    print(f"Overall reduction: {((total_old - total_new) / total_old * 100):.1f}%")
    print(f"\nDone! Saved to {guides_file}")
    print(f"Individual files saved to {individual_dir}")

if __name__ == '__main__':
    main()

