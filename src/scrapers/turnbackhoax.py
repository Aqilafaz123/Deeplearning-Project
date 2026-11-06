import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_turnbackhoax(category="all", max_pages=3, delay=0.3, start_page=1):
    """
    Scraper artikel TurnBackHoax (Mafindo).
    Mendukung kategori spesifik ('kesehatan', 'bantuan', 'lowongan') atau 'all' (semua topik).
    Label: 1 (Hoaks)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    collected_articles = []
    seen_urls = set()
    
    cat_label = "SEMUA TOPIK (Kesehatan, Penipuan, Bencana, dll.)" if category == "all" else f"'{category}'"
    print(f"\n[TurnBackHoax] Mengambil data hoaks ({cat_label}, Hal {start_page} s/d {start_page + max_pages - 1})...", flush=True)
    
    # 1. Kumpulkan daftar URL artikel dari halaman indeks
    article_urls = []
    for page in range(start_page, start_page + max_pages):
        if category == "all" or not category:
            index_url = f"https://turnbackhoax.id/articles?page={page}"
        else:
            index_url = f"https://turnbackhoax.id/articles?category={category}&page={page}"
            
        try:
            resp = session.get(index_url, timeout=10)
            if resp.status_code != 200:
                print(f"      [!] Gagal memuat indeks hal {page} (Status: {resp.status_code})", flush=True)
                continue
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            new_urls = 0
            for a in links:
                href = a.get('href', '')
                if '/articles/' in href and href != "https://turnbackhoax.id/articles" and href not in seen_urls:
                    seen_urls.add(href)
                    article_urls.append(href)
                    new_urls += 1
                    
            print(f"      Halaman {page}/{max_pages}: ditemukan {new_urls} tautan baru.", flush=True)
            if new_urls == 0:
                break
        except Exception as e:
            print(f"      [!] Error memuat indeks {page}: {e}", flush=True)

    print(f"      Total artikel hoaks yang akan diunduh: {len(article_urls)}", flush=True)
    if not article_urls:
        return pd.DataFrame()

    # 2. Unduh detail artikel dengan progress bar
    with tqdm(total=len(article_urls), desc="      Unduh TurnBackHoax", unit="art", file=sys.stdout) as pbar:
        for url in article_urls:
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
                
                # Metadata (Kategori artikel, Tanggal & Author)
                timestamp = ""
                author = "Mafindo"
                detected_cat = category if category != "all" else "Umum"
                
                p_meta = art_soup.select_one('article div.xl\\:col-span-2 p') or art_soup.find('p', class_=lambda c: c and 'gap-8' in c)
                if p_meta:
                    meta_text = p_meta.get_text(" ", strip=True)
                    parts = meta_text.split()
                    if len(parts) >= 1 and category == "all":
                        detected_cat = parts[0]
                    if len(parts) >= 2:
                        timestamp = parts[1]
                    if len(parts) >= 3:
                        author = parts[2]
                        
                # Konten Utama
                main_sec = art_soup.find('section', class_=lambda c: c and 'article--main' in c)
                if not main_sec:
                    main_sec = art_soup.find('article')
                    
                full_text = ""
                narasi = ""
                
                if main_sec:
                    full_text = main_sec.get_text(" ", strip=True)
                    text_lower = full_text.lower()
                    if "narasi" in text_lower:
                        paragraphs = [p.get_text(strip=True) for p in main_sec.find_all('p')]
                        for i, p in enumerate(paragraphs):
                            if "narasi" in p.lower() and i + 1 < len(paragraphs):
                                narasi = paragraphs[i + 1]
                                break
                            elif p.lower().startswith("narasi"):
                                narasi = p
                                break
                                
                if not narasi:
                    narasi = full_text[:1000]
                    
                collected_articles.append({
                    'Title': title,
                    'Timestamp': timestamp,
                    'FullText': full_text,
                    'Narasi': narasi,
                    'Tags': detected_cat,
                    'Author': author,
                    'Url': url,
                    'Category': detected_cat,
                    'hoax': 1
                })
            except Exception:
                pass
            finally:
                pbar.update(1)
                
    df = pd.DataFrame(collected_articles)
    print(f"      [OK] Selesai TurnBackHoax: {len(df)} berita hoaks terkumpul.\n", flush=True)
    return df
