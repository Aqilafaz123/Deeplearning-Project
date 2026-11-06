import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_kompas(category="all", max_pages=3, delay=0.3, start_page=1):
    """
    Scraper berita valid Kompas.com lintas rubrik.
    Mendukung rubrik spesifik atau 'all' (kesehatan, sains, edukasi, money, tren).
    Label: 0 (Valid / Fakta)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    collected_articles = []
    seen_urls = set()
    
    # Kanal Kompas
    if category == "kesehatan":
        channels = [("Kesehatan", "https://health.kompas.com/penyakit?page=")]
    else:  # category == 'all'
        channels = [
            ("Kesehatan", "https://health.kompas.com/penyakit?page="),
            ("Sains", "https://sains.kompas.com?page="),
            ("Edukasi", "https://edukasi.kompas.com?page="),
            ("Money", "https://money.kompas.com?page="),
            ("Tren", "https://tren.kompas.com?page=")
        ]
        
    print(f"\n[Kompas.com] Mengambil berita asli (Rubrik: {category.upper()}, Hal {start_page} s/d {start_page + max_pages - 1})...", flush=True)
    
    article_queue = []
    for cat_name, base_url in channels:
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
                    if '/read/' in href and 'kompas.com' in href and href not in seen_urls:
                        seen_urls.add(href)
                        article_queue.append((href, cat_name))
            except Exception:
                pass

    print(f"      Total artikel Kompas yang akan diunduh: {len(article_queue)}", flush=True)
    if not article_queue:
        return pd.DataFrame()

    with tqdm(total=len(article_queue), desc="      Unduh Kompas.com", unit="art", file=sys.stdout) as pbar:
        for url, cat_name in article_queue:
            time.sleep(delay)
            try:
                art_resp = session.get(url, timeout=10)
                if art_resp.status_code != 200:
                    pbar.update(1)
                    continue
                    
                art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                
                # Judul
                title_tag = art_soup.find('h1', class_='read__title') or art_soup.find('h1')
                title = title_tag.get_text(strip=True) if title_tag else ""
                if not title:
                    pbar.update(1)
                    continue
                    
                # Timestamp
                date_tag = art_soup.find('div', class_='read__time')
                timestamp = date_tag.get_text(strip=True) if date_tag else ""
                
                # Author
                author_tag = art_soup.find('div', class_='credit-title-name')
                author = author_tag.get_text(strip=True) if author_tag else "Kompas.com"
                
                # Konten Teks
                body_div = art_soup.find('div', class_='read__content')
                if body_div:
                    for unwanted in body_div(['script', 'style', 'iframe', 'ads']):
                        unwanted.decompose()
                    paragraphs = [p.get_text(strip=True) for p in body_div.find_all('p') if not p.find('strong')]
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
    print(f"      [OK] Selesai Kompas.com: {len(df)} berita asli terkumpul.\n", flush=True)
    return df
