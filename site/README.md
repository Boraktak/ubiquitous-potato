# Cerita — situs cerita bergaya iOS

Aplikasi web mandiri (standalone) untuk membaca & menulis cerita, dengan
tampilan bersih bergaya iOS. Folder ini **tidak bergantung** pada proyek ML
di root repository — keduanya terpisah sepenuhnya.

## Menjalankan

Cukup Python 3 (tanpa package tambahan):

```bash
cd site
python3 server.py        # berjalan di http://0.0.0.0:8080
```

`server.py` melayani file statis **dan** penyimpanan cerita lewat API kecil:

| Method | Path | Fungsi |
|---|---|---|
| `GET` | `/api/stories` | ambil semua cerita (JSON) |
| `PUT` | `/api/stories` | timpa seluruh daftar cerita (body = array JSON) |

Cerita tersimpan di `stories.json` (dibuat otomatis, sudah di-`.gitignore`).
Jika server tidak terjangkau, aplikasi otomatis jatuh ke `localStorage`
(badge **Offline** muncul) sehingga tetap bisa dipakai.

Tanpa server pun tetap jalan: buka `index.html` langsung di browser, dan semua
data disimpan di `localStorage` perangkat.

## Isi folder

| File | Fungsi |
|---|---|
| `index.html` | struktur halaman |
| `styles.css` | desain iOS + tema terang/gelap |
| `app.js` | seluruh logika (tanpa dependency) |
| `server.py` | server statis + penyimpanan JSON (stdlib only) |

## Fitur

- Baca & tulis cerita (judul, penulis, 7 kategori, isi)
- **Edit** cerita, **hapus** dengan konfirmasi + undo
- **Favorit** (suka) + filter Semua / Terbaru / Populer / Favorit
- **Bagikan** (Web Share API, fallback salin teks)
- Tema **Terang / Gelap** (mengikuti sistem, bisa ditimpa, tersimpan)
- Pencarian, progress baca, toast notifikasi, aksesibilitas
  (`prefers-reduced-motion`, fokus keyboard, escape untuk menutup sheet)

## Memisahkan dari repo induk

Jika ingin memindahkan situs ini ke repository sendiri (mis. `cerita`):

```bash
# dari root repository
git subtree split --prefix=site -b site-only
gh repo create Boraktak/cerita --public --source=site-only --push
```
