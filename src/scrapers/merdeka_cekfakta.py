import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_merdeka_cekfakta(max_pages=3, delay=0.3, start_page=1):
    """
    Scraper Merdeka.com Cek Fakta.
    Kanal klarifikasi hoaks dan disinformasi.
    Label: 1 (Hoaks)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    collected_articles = []
    seen_urls = set()
    
    print(f"\n[Merdeka Cek Fakta] Mengambil data hoaks (Hal {start_page} s/d {start_page + max_pages - 1})...", flush=True)
    
    article_urls = []
    for page in range(start_page, start_page + max_pages):
        url = f"https://www.merdeka.com/cek-fakta/?page={page}"
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', href=True)
            for a in links:
                href = a['href']
                if '/cek-fakta/' in href and ('read' in href or len(href.split('/')) > 4):
                    full_url = "https://www.merdeka.com" + href if href.startswith('/') else href
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        article_urls.append(full_url)
        except Exception as e:
            print(f"      [!] Error indeks Merdeka hal {page}: {e}", flush=True)

    print(f"      Total artikel Merdeka Cek Fakta ditemukan: {len(article_urls)}", flush=True)
    if not article_urls:
        return pd.DataFrame()

    with tqdm(total=len(article_urls), desc="      Unduh Merdeka Cek Fakta", unit="art", file=sys.stdout) as pbar:
        for url in article_urls:
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
                    art_soup.find('div', class_='articles-content') or
                    art_soup.find('div', class_='mdk-body-paragraph') or
                    art_soup.find('div', class_='content-text') or
                    art_soup.find('article')
                )
                paragraphs = [p.get_text(strip=True) for p in body_div.find_all('p')] if body_div else []
                full_text = " ".join([p for p in paragraphs if p and not p.startswith("BACA JUGA")])
                
                if len(full_text) < 50:
                    pbar.update(1)
                    continue
                    
                date_tag = art_soup.find('time') or art_soup.find('span', class_=lambda c: c and 'date' in c)
                timestamp = date_tag.get_text(strip=True) if date_tag else ""
                
                collected_articles.append({
                    'Title': title,
                    'Timestamp': timestamp,
                    'FullText': full_text,
                    'Narasi': title,
                    'Tags': 'Cek Fakta, Hoaks, Merdeka',
                    'Author': 'Merdeka.com',
                    'Url': url,
                    'Category': 'Cek Fakta',
                    'hoax': 1
                })
            except Exception:
                pass
            pbar.update(1)

    print(f"   [V] Berhasil mengunduh {len(collected_articles)} artikel dari Merdeka Cek Fakta.", flush=True)
    return pd.DataFrame(collected_articles)

