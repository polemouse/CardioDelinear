# scripts/download_datasets.py
import os
import wfdb
from wfdb import dl_database

os.makedirs('data/ludb', exist_ok=True)
os.makedirs('data/qtdb', exist_ok=True)

# LUDB (Lobachevsky University Database) — short 10s, 12-lead
# PhysioNet slug often: 'ludb'
dl_database('ludb', dl_dir='data/ludb')

# QT Database — 15-min excerpts, 2-lead; partial expert annotations
# PhysioNet slug: 'qtdb'
dl_database('qtdb', dl_dir='data/qtdb')

print('Done. LUDB → data/ludb, QTDB → data/qtdb')