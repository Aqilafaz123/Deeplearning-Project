import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_liputan6_cekfakta(max_pages=3, delay=0.3, start_page=1):
    """
    Scraper Liputan6 Cek Fakta.
    Platform cek fakta terakreditasi IFCN.
    Label: 1 (Hoaks / Cek Fakta)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    collected_articles = []
    seen_urls = set()
    
    print(f"\n[Liputan6 Cek Fakta] Mengambil data hoaks (Hal {start_page} s/d {start_page + max_pages - 1})...", flush=True)
    
    article_urls = []
    for page in range(start_page, start_page + max_pages):
        url = f"https://www.liputan6.com/cek-fakta?page={page}"
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', href=True)
            for a in links:
                href = a['href']
                if '/cek-fakta/read/' in href and href not in seen_urls:
                    seen_urls.add(href)
                    article_urls.append(href)
        except Exception:
            pass

    print(f"      Total artikel Liputan6 yang ditemukan: {len(article_urls)}", flush=True)
    if not article_urls:
        return pd.DataFrame()

    with tqdm(total=len(article_urls), desc="      Unduh Liputan6", unit="art", file=sys.stdout) as pbar:
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
                    
                body = art_soup.find('div', class_='article-content-body__item-content') or art_soup.find('div', class_=lambda c: c and 'article-content' in c)
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
                    'Narasi': full_text[:1000],
                    'Tags': 'Cek Fakta',
                    'Author': 'Liputan6 Cek Fakta',
                    'Url': url,
                    'Category': 'Cek Fakta / Hoaks',
                    'hoax': 1
                })
            except Exception:
                pass
            finally:
                pbar.update(1)
                
    df = pd.DataFrame(collected_articles)
    print(f"      [OK] Selesai Liputan6: {len(df)} berita hoaks terkumpul.\n", flush=True)
    return df


if __name__ == "__main__":
    df = scrape_liputan6_cekfakta(max_pages=1, delay=0.2)
    print("Liputan6 sample len:", len(df))
    if not df.empty:
        print(df[['Title', 'hoax']].head(3))

