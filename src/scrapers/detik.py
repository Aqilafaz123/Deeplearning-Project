import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_detik(category="all", max_pages=3, delay=0.3, start_page=1):
    """
    Scraper portal berita valid Detik.com lintas kanal.
    Mendukung kanal spesifik atau 'all' (kesehatan, teknologi/inet, edukasi, finance, news).
    Label: 0 (Fakta / Valid)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    collected_articles = []
    seen_urls = set()
    
    if category == "kesehatan":
        channels = [("Kesehatan", "https://health.detik.com/indeks?page=")]
    elif category == "tekno" or category == "teknologi":
        channels = [("Teknologi", "https://inet.detik.com/indeks?page=")]
    elif category == "ekonomi":
        channels = [("Ekonomi", "https://finance.detik.com/indeks?page=")]
    elif category == "edukasi":
        channels = [("Edukasi", "https://www.detik.com/edu/indeks?page=")]
    else:  # all
        channels = [
            ("Nasional", "https://news.detik.com/indeks?page="),
            ("Kesehatan", "https://health.detik.com/indeks?page="),
            ("Teknologi", "https://inet.detik.com/indeks?page="),
            ("Ekonomi", "https://finance.detik.com/indeks?page="),
            ("Edukasi", "https://www.detik.com/edu/indeks?page=")
        ]
        
    print(f"\n[Detik.com] Mengambil berita asli (Rubrik: {category.upper()}, Hal {start_page} s/d {start_page + max_pages - 1})...", flush=True)
    
    article_queue = []
    for cat_name, base_url in channels:
        pages_to_scrape = max(1, max_pages // len(channels)) if category == "all" else max_pages
        for page in range(start_page, start_page + pages_to_scrape):
            url = f"{base_url}{page}"
            try:
                resp = session.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, 'html.parser')
                links = soup.find_all('a', href=True)
                for a in links:
                    href = a['href']
                    if '/d-' in href and 'detik.com' in href and href not in seen_urls:
                        seen_urls.add(href)
                        article_queue.append((href, cat_name))
            except Exception as e:
                print(f"      [!] Error indeks Detik {cat_name} hal {page}: {e}", flush=True)

    print(f"      Total artikel Detik.com yang akan diunduh: {len(article_queue)}", flush=True)
    if not article_queue:
        return pd.DataFrame()

    with tqdm(total=len(article_queue), desc="      Unduh Detik.com", unit="art", file=sys.stdout) as pbar:
        for url, cat_name in article_queue:
            time.sleep(delay)
            try:
                art_resp = session.get(url, timeout=10)
                if art_resp.status_code != 200:
                    pbar.update(1)
                    continue
                art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                
                title_tag = art_soup.find('h1')
                title = title_tag.get_text(strip=True) if title_tag else ""
                if not title:
                    pbar.update(1)
                    continue
                    
                body_div = (
                    art_soup.find('div', class_='detail__body-text') or
                    art_soup.find('div', class_='itp_bodycontent') or
                    art_soup.find('div', class_=lambda c: c and 'detail__body' in c)
                )
                
                paragraphs = [p.get_text(strip=True) for p in body_div.find_all('p')] if body_div else []
                full_text = " ".join([p for p in paragraphs if p and not p.startswith("ADVERTISEMENT") and not p.startswith("SCROLL TO CONTINUE")])
                
                if len(full_text) < 50:
                    pbar.update(1)
                    continue
                    
                date_tag = art_soup.find('div', class_='detail__date') or art_soup.find('div', class_='date')
                timestamp = date_tag.get_text(strip=True) if date_tag else ""
                
                author_tag = art_soup.find('div', class_='detail__author')
                author = author_tag.get_text(strip=True) if author_tag else "detikcom"
                
                collected_articles.append({
                    'Title': title,
                    'Timestamp': timestamp,
                    'FullText': full_text,
                    'Tags': f"Detik, {cat_name}",
                    'Author': author,
                    'Url': url,
                    'Category': cat_name,
                    'hoax': 0
                })
            except Exception:
                pass
            pbar.update(1)

    print(f"   [V] Berhasil mengunduh {len(collected_articles)} artikel dari Detik.com.", flush=True)
    return pd.DataFrame(collected_articles)

