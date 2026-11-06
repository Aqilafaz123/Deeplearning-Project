import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_antara(category="all", max_pages=3, delay=0.3, start_page=1):
    """
    Scraper Antara News (Kantor Berita Nasional Indonesia).
    Label: 0 (Valid / Fakta)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    collected_articles = []
    seen_urls = set()
    
    if category == "kesehatan":
        channels = [("Kesehatan", "https://www.antaranews.com/warta-berita/kesehatan")]
    elif category == "tekno":
        channels = [("Tekno", "https://www.antaranews.com/tekno")]
    elif category == "ekonomi":
        channels = [("Ekonomi", "https://www.antaranews.com/ekonomi")]
    else:
        channels = [
            ("Umum", "https://www.antaranews.com/terkini/"),
            ("Tekno", "https://www.antaranews.com/tekno/"),
            ("Ekonomi", "https://www.antaranews.com/ekonomi/")
        ]
        
    print(f"\n[Antara News] Mengambil berita asli multi-rubrik (Hal {start_page} s/d {start_page + max_pages - 1})...", flush=True)
    
    article_queue = []
    for cat_name, base_url in channels:
        for page in range(start_page, start_page + max_pages):
            url = f"{base_url}{page}" if base_url.endswith('/') else f"{base_url}/{page}"
            try:
                resp = session.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, 'html.parser')
                links = soup.find_all('a', href=True)
                for a in links:
                    href = a['href']
                    if '/berita/' in href and href not in seen_urls:
                        seen_urls.add(href)
                        article_queue.append((href, cat_name))
            except Exception:
                pass

    print(f"      Total artikel Antara yang ditemukan: {len(article_queue)}", flush=True)
    if not article_queue:
        return pd.DataFrame()

    with tqdm(total=len(article_queue), desc="      Unduh Antara News", unit="art", file=sys.stdout) as pbar:
        for url, cat_name in article_queue:
            time.sleep(delay)
            try:
                art_resp = session.get(url, timeout=10)
                if art_resp.status_code != 200:
                    pbar.update(1)
                    continue
                art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                
                title_tag = art_soup.find('h1', class_='post-title') or art_soup.find('h1')
                title = title_tag.get_text(strip=True) if title_tag else ""
                if not title:
                    pbar.update(1)
                    continue
                    
                body = art_soup.find('div', class_='post-content')
                if body:
                    for s in body(['script', 'style', 'iframe', 'ads']):
                        s.decompose()
                    paragraphs = [p.get_text(strip=True) for p in body.find_all('p')]
                    full_text = " ".join([p for p in paragraphs if len(p) > 20])
                else:
                    paragraphs = [p.get_text(strip=True) for p in art_soup.find_all('p')]
                    full_text = " ".join([p for p in paragraphs if len(p) > 20])
                    
                if not full_text:
                    pbar.update(1)
                    continue
                    
                collected_articles.append({
                    'Title': title,
                    'Timestamp': "",
                    'FullText': full_text,
                    'Tags': cat_name,
                    'Author': 'Antara News',
                    'Url': url,
                    'Category': cat_name,
                    'hoax': 0
                })
            except Exception:
                pass
            finally:
                pbar.update(1)
                
    df = pd.DataFrame(collected_articles)
    print(f"      [OK] Selesai Antara: {len(df)} berita asli terkumpul.\n", flush=True)
    return df

