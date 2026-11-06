import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_cnn(category="all", max_pages=3, delay=0.3, start_page=1):
    """
    Scraper berita valid CNN Indonesia lintas rubrik.
    Mendukung rubrik spesifik atau 'all' (campuran kesehatan, teknologi, edukasi, ekonomi, gaya-hidup).
    Label: 0 (Valid / Fakta)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    collected_articles = []
    seen_urls = set()
    
    # Tentukan channel yang akan di-scrape
    if category == "kesehatan":
        channels = [("kesehatan", "https://www.cnnindonesia.com/gaya-hidup/kesehatan?page=")]
    elif category == "teknologi":
        channels = [("teknologi", "https://www.cnnindonesia.com/teknologi?page=")]
    elif category == "edukasi":
        channels = [("edukasi", "https://www.cnnindonesia.com/edukasi?page=")]
    elif category == "ekonomi":
        channels = [("ekonomi", "https://www.cnnindonesia.com/ekonomi?page=")]
    else:  # category == 'all'
        channels = [
            ("Kesehatan", "https://www.cnnindonesia.com/gaya-hidup/kesehatan?page="),
            ("Teknologi", "https://www.cnnindonesia.com/teknologi?page="),
            ("Edukasi", "https://www.cnnindonesia.com/edukasi?page="),
            ("Ekonomi", "https://www.cnnindonesia.com/ekonomi?page="),
            ("Nasional", "https://www.cnnindonesia.com/nasional?page=")
        ]
        
    print(f"\n[CNN Indonesia] Mengambil berita asli (Rubrik: {category.upper()}, Hal {start_page} s/d {start_page + max_pages - 1})...", flush=True)
    
    article_queue = []
    # Kumpulkan tautan
    for cat_name, base_url in channels:
        # Jika 'all', bagi halaman per kanal agar seimbang
        pages_to_scrape = max(1, max_pages // len(channels)) if category == "all" else max_pages
        for page in range(start_page, start_page + pages_to_scrape):
            index_url = f"{base_url}{page}"
            try:
                resp = session.get(index_url, timeout=10)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, 'html.parser')
                links = soup.find_all('a', href=True)
                for a in links:
                    href = a['href']
                    if 'cnnindonesia.com/' in href and len(href.split('/')) > 5 and href not in seen_urls:
                        seen_urls.add(href)
                        article_queue.append((href, cat_name))
            except Exception:
                pass

    print(f"      Total artikel CNN yang akan diunduh: {len(article_queue)}", flush=True)
    if not article_queue:
        return pd.DataFrame()

    with tqdm(total=len(article_queue), desc="      Unduh CNN Indonesia", unit="art", file=sys.stdout) as pbar:
        for url, cat_name in article_queue:
            time.sleep(delay)
            try:
                art_resp = session.get(url, timeout=10)
                if art_resp.status_code != 200:
                    pbar.update(1)
                    continue
                    
                art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                
                # Judul
                title_tag = art_soup.find('h1')
                title = title_tag.get_text(strip=True) if title_tag else ""
                if not title:
                    pbar.update(1)
                    continue
                    
                # Timestamp
                date_tag = art_soup.find('div', class_=lambda c: c and 'date' in c) or art_soup.select_one('.text-cnn_grey')
                timestamp = date_tag.get_text(strip=True) if date_tag else ""
                
                # Author
                author_tag = art_soup.find('div', class_=lambda c: c and 'author' in c) or art_soup.select_one('.text-cnn_red')
                author = author_tag.get_text(strip=True) if author_tag else "CNN Indonesia"
                
                # Konten Teks
                body_div = art_soup.find('div', class_=lambda c: c and 'detail-text' in c) or art_soup.select_one('#detikdetailtext')
                if body_div:
                    for unwanted in body_div(['script', 'style', 'iframe', 'table', 'aside']):
                        unwanted.decompose()
                    paragraphs = [p.get_text(strip=True) for p in body_div.find_all('p')]
                    full_text = " ".join([p for p in paragraphs if len(p) > 20])
                else:
                    paragraphs = [p.get_text(strip=True) for p in art_soup.find_all('p')]
                    full_text = " ".join([p for p in paragraphs if len(p) > 20])
                    
                if not full_text:
                    pbar.update(1)
                    continue
                    
                collected_articles.append({
                    'Title': title,
                    'Timestamp': timestamp,
                    'FullText': full_text,
                    'Tags': cat_name,
                    'Author': author,
                    'Url': url,
                    'Category': cat_name,
                    'hoax': 0
                })
            except Exception:
                pass
            finally:
                pbar.update(1)
                
    df = pd.DataFrame(collected_articles)
    print(f"      [OK] Selesai CNN Indonesia: {len(df)} berita asli terkumpul.\n", flush=True)
    return df
