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

## Deploy ke GitHub Pages (web publik)

Catatan penting: Pages bersifat **statis**, jadi cerita hanya tersimpan di
`localStorage` per browser (badge **Offline** akan tampil — itu normal).
Untuk penyimpanan multi-perangkat tetap pakai `python3 server.py`.

Langkah satu kali untuk mengaktifkan (butuh akses **admin** repo):

1. Buat file `.github/workflows/deploy-pages.yml` di repo (isi lihat di bawah).
2. Buka **Settings → Pages**, pada *Build and deployment* pilih
   **Source: GitHub Actions**.
3. Setelah itu workflow berjalan otomatis tiap push ke `main` (atau manual
   lewat tab **Actions → Run workflow**).
4. URL situs: `https://boraktak.github.io/ubiquitous-potato/`

Isi `.github/workflows/deploy-pages.yml`:

```yaml
name: Deploy Cerita site to GitHub Pages

on:
  push:
    branches: ["main", "arena/01a00d61-ubiquitous-potato"]
  workflow_dispatch:

concurrency:
  group: pages
  cancel-in-progress: true

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - name: Build site (copy site/ to _site)
        run: |
          mkdir -p _site
          cp -r site/. _site/
      - uses: actions/upload-pages-artifact@v3
        with:
          path: _site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

> Mengapa langkah di atas manual? Token bot yang dipakai Arena tidak punya
> izin `admin` (untuk mengaktifkan Pages) maupun `workflows` (untuk mengunggah
> file workflow), jadi dua langkah di atas dilakukan oleh pemilik repo. Setelah
> Pages aktif, deploy otomatis.

## Memisahkan dari repo induk

Jika ingin memindahkan situs ini ke repository sendiri (mis. `cerita`):

```bash
# dari root repository
git subtree split --prefix=site -b site-only
gh repo create Boraktak/cerita --public --source=site-only --push
```
