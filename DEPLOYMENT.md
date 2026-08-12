# Deploy HARNESS untuk pilot 3–5 orang

## Prasyarat wajib
1. Buat PostgreSQL terkelola (Neon/Supabase/Railway/Render).
2. Deploy aplikasi Next.js ke host publik.
3. Jangan memakai `127.0.0.1` untuk `DATABASE_URL` production.
4. Isi `APP_BASE_URL` untuk production. Agent Runner menolak berjalan tanpa ini.

## Environment variables
Salin `.env.example` lalu isi minimal:
- `DATABASE_URL`
- `APP_BASE_URL`
- `PILOT_ACCESS_KEY`
- `PILOT_USERNAME`
- `OPENAI_API_KEY` (opsional, tetapi disarankan)
- `UPSTASH_REDIS_REST_URL` dan `UPSTASH_REDIS_REST_TOKEN` (opsional untuk rate limit bersama)

## Database
```bash
npm install
npm run db:migrate
```

## Build & deploy
```bash
npm run typecheck
npm run lint
npm run build
```

## Setelah deploy
1. Buka `/api/health`.
2. Buka halaman utama; browser harus meminta Basic Auth jika `PILOT_ACCESS_KEY` terpasang.
3. Buat satu project dummy, konfirmasi, pilih satu jalur eksekusi.
4. Uji preview dari perangkat lain.
5. Submit form asli sekali dan pastikan lead bertambah satu.
6. Jalankan Verifier dan pastikan lead tidak bertambah (submission verifier disimpan sebagai `is_test=true`).

## Batas keamanan pilot
- Preview publik agar bisa dibagikan.
- Metadata project, daftar project, leads, dan endpoint mutasi dilindungi Basic Auth.
- Rate limit mendukung in-memory fallback dan opsional Upstash Redis.
- Untuk produksi multi-user, ganti Basic Auth dengan akun per-user, session, ownership, CSRF token, dan audit log yang lebih serius.
