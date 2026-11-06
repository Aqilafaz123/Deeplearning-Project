"""
Script Pengambil Dataset Berita Berkelanjutan (Continuous Scraper) Multi-Media
Mendukung 7 Portal Berita:
  1. Hoaks (Label 1):
     - TurnBackHoax (Mafindo)
     - Liputan6 Cek Fakta (IFCN)
     - Merdeka Cek Fakta
  2. Fakta / Berita Asli (Label 0):
     - Detik.com (Nasional, Kesehatan, Inet/Tekno, Edukasi, Finance)
     - Kompas.com (Edukasi, Sains, Kesehatan, Money, Tren)
     - CNN Indonesia (Nasional, Kesehatan, Teknologi, Edukasi, Ekonomi)
     - Antara News (Kantor Berita Nasional)
"""

import os
import sys
import time
import json
import signal
import argparse
import pandas as pd
from datetime import datetime

# Pastikan output terminal tidak ter-buffer
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from scrape_dataset import safe_save, save_or_append_df
from src.scrapers import (
    scrape_turnbackhoax,
    scrape_liputan6_cekfakta,
    scrape_merdeka_cekfakta,
    scrape_detik,
    scrape_kompas,
    scrape_cnn,
    scrape_antara
)

CHECKPOINT_FILE = "dataset_multimedia/RAW/scraper_checkpoint.json"


def load_checkpoint():
    """Memuat posisi halaman terakhir dari tiap sumber media."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "turnbackhoax": 1,
        "liputan6": 1,
        "merdeka": 1,
        "detik": 1,
        "kompas": 1,
        "antara": 1,
        "cnn": 1,
        "cycles_completed": 0
    }


def save_checkpoint(state):
    """Menyimpan posisi halaman terakhir ke file checkpoint."""
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"   [!] Gagal simpan checkpoint: {e}", flush=True)


def get_current_total(output_dir="dataset_multimedia/RAW"):
    """Menghitung total baris saat ini di file raw."""
    total = 0
    raw_files = [
        "dataset_turnbackhoax_raw.xlsx",
        "dataset_liputan6_raw.xlsx",
        "dataset_merdeka_raw.xlsx",
        "dataset_detik_raw.xlsx",
        "dataset_kompas_raw.xlsx",
        "dataset_antara_raw.xlsx",
        "dataset_cnn_raw.xlsx"
    ]
    for fname in raw_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_excel(fpath)
                total += len(df)
            except Exception:
                pass
    return total


def update_combined_dataset(output_dir="dataset_multimedia/RAW"):
    """Menggabungkan seluruh file raw individual menjadi dataset gabungan terpadu."""
    raw_files = [
        ("dataset_turnbackhoax_raw.xlsx", "TurnBackHoax", 1),
        ("dataset_liputan6_raw.xlsx", "Liputan6", 1),
        ("dataset_merdeka_raw.xlsx", "Merdeka", 1),
        ("dataset_detik_raw.xlsx", "Detik", 0),
        ("dataset_kompas_raw.xlsx", "Kompas", 0),
        ("dataset_antara_raw.xlsx", "Antara", 0),
        ("dataset_cnn_raw.xlsx", "CNN", 0),
    ]
    dfs = []
    for fname, source, label in raw_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_excel(fpath)
                df['Source'] = source
                if 'hoax' not in df.columns:
                    df['hoax'] = label
                dfs.append(df)
            except Exception:
                pass
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        subset = [c for c in ['Url', 'Title'] if c in merged.columns]
        if subset:
            merged = merged.drop_duplicates(subset=subset).reset_index(drop=True)
        c_xlsx = os.path.join(output_dir, "dataset_multimedia_combined.xlsx")
        c_csv = os.path.join(output_dir, "dataset_multimedia_combined.csv")
        safe_save(merged, c_xlsx)
        safe_save(merged, c_csv)
        return len(merged)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Continuous / Infinite News Scraper (7 Media Sources)")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "turnbackhoax", "liputan6", "merdeka", "detik", "cnn", "kompas", "antara"],
        help="Sumber berita yang ingin diambil terus-menerus (default: all)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Kategori berita (default: all)"
    )
    parser.add_argument(
        "--batch_pages",
        type=int,
        default=5,
        help="Jumlah halaman per siklus per media (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Jeda per artikel dalam detik (default: 0.4)"
    )
    parser.add_argument(
        "--cycle_delay",
        type=int,
        default=3,
        help="Jeda istirahat antar siklus dalam detik (default: 3)"
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=0,
        help="Berhenti otomatis jika total artikel mencapai angka ini (0 = jalan tanpa batas)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_multimedia/RAW",
        help="Folder output dataset (default: dataset_multimedia/RAW)"
    )
    parser.add_argument(
        "--reset_checkpoint",
        action="store_true",
        help="Reset nomor halaman awal kembali ke 1"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    state = load_checkpoint() if not args.reset_checkpoint else {
        "turnbackhoax": 1,
        "liputan6": 1,
        "merdeka": 1,
        "detik": 1,
        "kompas": 1,
        "antara": 1,
        "cnn": 1,
        "cycles_completed": 0
    }

    print("\n" + "=" * 70, flush=True)
    print(" CONTINUOUS MULTI-MEDIA NEWS SCRAPER (7 MEDIA TERKEMUKA INDONESIA)", flush=True)
    print("=" * 70, flush=True)
    print(" [Sumber Hoaks] : TurnBackHoax (Mafindo), Liputan6 Cek Fakta, Merdeka Cek Fakta", flush=True)
    print(" [Sumber Fakta] : Detik.com, Kompas.com, CNN Indonesia, Antara News", flush=True)
    print("-" * 70, flush=True)
    print(f" Target Media       : {args.source.upper()}", flush=True)
    print(f" Halaman per Siklus : {args.batch_pages} hal per media", flush=True)
    print(f" Jeda per Artikel   : {args.delay} detik", flush=True)
    print(f" Target Total Data  : {f'{args.target_total:,} artikel' if args.target_total > 0 else 'TANPA BATAS (Non-Stop)'}", flush=True)
    print(" PETUNJUK: Tekan [Ctrl + C] kapan saja untuk BERHENTI DENGAN AMAN.", flush=True)
    print("           Data yang sudah didownload akan langsung tersimpan.", flush=True)
    print("=" * 70 + "\n", flush=True)

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n\n[!] Sinyal berhenti diterima (Ctrl+C). Menyelesaikan penyimpanan terakhir...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    cycle = state.get("cycles_completed", 0)

    try:
        while running:
            cycle += 1
            print(f"\n>>> [SIKLUS {cycle}] Memulai scraping batch halaman...", flush=True)

            # 1. TurnBackHoax (Hoaks)
            if running and args.source in ["all", "turnbackhoax"]:
                p_start = state.get("turnbackhoax", 1)
                try:
                    df = scrape_turnbackhoax(category=args.category, max_pages=args.batch_pages, delay=args.delay, start_page=p_start)
                    if not df.empty:
                        save_or_append_df(df, os.path.join(args.output_dir, "dataset_turnbackhoax_raw.xlsx"))
                    state["turnbackhoax"] = p_start + args.batch_pages
                    save_checkpoint(state)
                except Exception as e:
                    print(f"   [!] Error TurnBackHoax: {e}", flush=True)

            # 2. Liputan6 Cek Fakta (Hoaks)
            if running and args.source in ["all", "liputan6"] and state.get("liputan6", 1) <= 15:
                p_start = state.get("liputan6", 1)
                try:
                    df = scrape_liputan6_cekfakta(max_pages=args.batch_pages, delay=args.delay, start_page=p_start)
                    if not df.empty:
                        save_or_append_df(df, os.path.join(args.output_dir, "dataset_liputan6_raw.xlsx"))
                    state["liputan6"] = p_start + args.batch_pages
                    save_checkpoint(state)
                except Exception as e:
                    print(f"   [!] Error Liputan6: {e}", flush=True)

            # 3. Merdeka Cek Fakta (Hoaks)
            if running and args.source in ["all", "merdeka"] and state.get("merdeka", 1) <= 4:
                p_start = state.get("merdeka", 1)
                try:
                    df = scrape_merdeka_cekfakta(max_pages=args.batch_pages, delay=args.delay, start_page=p_start)
                    if not df.empty:
                        save_or_append_df(df, os.path.join(args.output_dir, "dataset_merdeka_raw.xlsx"))
                    state["merdeka"] = p_start + args.batch_pages
                    save_checkpoint(state)
                except Exception as e:
                    print(f"   [!] Error Merdeka Cek Fakta: {e}", flush=True)

            # 4. Detik.com (Fakta)
            if running and args.source in ["all", "detik"]:
                p_start = state.get("detik", 1)
                try:
                    df = scrape_detik(category=args.category, max_pages=args.batch_pages, delay=args.delay, start_page=p_start)
                    if not df.empty:
                        save_or_append_df(df, os.path.join(args.output_dir, "dataset_detik_raw.xlsx"))
                    state["detik"] = p_start + args.batch_pages
                    save_checkpoint(state)
                except Exception as e:
                    print(f"   [!] Error Detik.com: {e}", flush=True)

            # 5. Kompas (Fakta)
            if running and args.source in ["all", "kompas"]:
                p_start = state.get("kompas", 1)
                try:
                    df = scrape_kompas(category=args.category, max_pages=args.batch_pages, delay=args.delay, start_page=p_start)
                    if not df.empty:
                        save_or_append_df(df, os.path.join(args.output_dir, "dataset_kompas_raw.xlsx"))
                    state["kompas"] = p_start + args.batch_pages
                    save_checkpoint(state)
                except Exception as e:
                    print(f"   [!] Error Kompas: {e}", flush=True)

            # 6. Antara News (Fakta)
            if running and args.source in ["all", "antara"]:
                p_start = state.get("antara", 1)
                try:
                    df = scrape_antara(category=args.category, max_pages=args.batch_pages, delay=args.delay, start_page=p_start)
                    if not df.empty:
                        save_or_append_df(df, os.path.join(args.output_dir, "dataset_antara_raw.xlsx"))
                    state["antara"] = p_start + args.batch_pages
                    save_checkpoint(state)
                except Exception as e:
                    print(f"   [!] Error Antara: {e}", flush=True)

            # 7. CNN Indonesia (Fakta)
            if running and args.source in ["all", "cnn"]:
                p_start = state.get("cnn", 1)
                try:
                    df = scrape_cnn(category=args.category, max_pages=args.batch_pages, delay=args.delay, start_page=p_start)
                    if not df.empty:
                        save_or_append_df(df, os.path.join(args.output_dir, "dataset_cnn_raw.xlsx"))
                    state["cnn"] = p_start + args.batch_pages
                    save_checkpoint(state)
                except Exception as e:
                    print(f"   [!] Error CNN: {e}", flush=True)

            state["cycles_completed"] = cycle
            save_checkpoint(state)

            current_total = get_current_total(args.output_dir)
            combined_count = update_combined_dataset(args.output_dir)
            print(f"\n[INFO SIKLUS {cycle} SELESAI] Total Terkumpul: ~{current_total:,} baris | Unik Gabungan: {combined_count:,} artikel.", flush=True)

            if args.target_total > 0 and current_total >= args.target_total:
                print(f"\n[TARGET TERCAPAI!] Total data telah mencapai target {current_total:,} >= {args.target_total:,}.", flush=True)
                break

            if running:
                print(f"Istirahat {args.cycle_delay} detik sebelum siklus berikutnya... (Ctrl+C untuk berhenti)", flush=True)
                time.sleep(args.cycle_delay)

    except KeyboardInterrupt:
        print("\n\n[!] Dihentikan oleh pengguna (Ctrl+C).", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("                  HASIL AKHIR SCRAPING BERKELANJUTAN", flush=True)
    print("=" * 70, flush=True)
    final_total = get_current_total(args.output_dir)
    final_combined = update_combined_dataset(args.output_dir)
    print(f" Total siklus berjalan : {cycle} siklus", flush=True)
    print(f" Total baris tersimpan : {final_total:,} artikel", flush=True)
    print(f" Total unik gabungan   : {final_combined:,} artikel", flush=True)
    print(f" Folder penyimpanan    : {os.path.abspath(args.output_dir)}", flush=True)
    print(f" Checkpoint tersimpan  : {CHECKPOINT_FILE}", flush=True)
    print(" Anda bisa melanjutkan scraping kapan saja dengan menjalankan script yang sama.", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
