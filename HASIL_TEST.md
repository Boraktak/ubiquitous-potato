# Hasil Test — DialecticTransformer (episodic memory + chunkwise SSM)

Tanggal: 2026-08-16
Lingkungan: Python 3.11.2, PyTorch 2.13.0+cu130 (CPU, CUDA tidak tersedia)
Deterministik: `torch.manual_seed(42)`, 3x run identik → tidak flaky.

## Hasil akhir: SEMUA OK (21/21)

```
Device  : cpu
Menjalankan A1 + episodic (chunk-invariant, CE only)...

Loss & Mem stats:
   - Loss/CE                : 230.9024
   - Loss/Total             : 230.9024
   - Mem/L0_opens           : 8.000000
   - Mem/L0_read_norm       : 1.218787
   - Mem/L0_inject_norm     : 0.001219
   - Mem/L0_valid_slots     : 3.500000
   ... (L1–L3 sama, opens=8, valid_slots=3.5)

Gradien:
   - SSM drift           : OK
   - Memory out          : OK
   - Mem inject          : OK
   - Query/Key/Value     : OK
   - Carry + 2nd backward: OK

Konsistensi:
   - State carry (mem)   : OK
   - No KV in training   : OK
   - S=0                 : OK
   - ignore_index        : OK
   - generate            : OK
   - Split mid-chunk     : OK
   - Split chunk-bound   : OK
   - SSM-only split (nocache): OK
   - Cache split         : OK
   - Pad-id independence : OK
   - Eviction FIFO       : OK

Ablasi episodic:
   - Retrieve live/zero  : OK
   - Zero-mem max|d|     : 1.924e-03 -> OK
   - Shuffle-mem max|d|  : 2.335e-03 -> OK

Hasil akhir: SEMUA OK
```

## Temuan saat test pertama (skrip asli: 19/21, 2 gagal)

Dua check gagal pada versi asli: **Split mid-chunk** dan **Split chunk-bound**.

Penyebab (terverifikasi lewat diagnostik):

| Uji | diff maks (asli) | diff maks (setelah fix) |
|---|---|---|
| SSM saja, split di 8, tanpa cache | 0.0 / 3.6e-7 | (sama, sudah invariant) |
| Full model, split di 8, tanpa cache | **22.48** | — |
| Full model, split di 8, pakai cache | 6.1e-5 | 6.1e-5 |

Akar masalahnya **bukan** di jalur chunk-invariant (SSM). SSM (`ChunkwiseSSM`)
terbukti chunk-invariant tanpa cache (diff ≈ 0). Yang bikin gagal adalah cabang
**causal attention**: saat forward penuh 256 token, token ke-8 dst. melihat
seluruh history; saat di-split tanpa cache, tiap potongan cuma melihat dirinya
sendiri (attention lokal). Jadi `full[:, 8:] != post` sebesar ~22.5.

Perbaikan di `dialectic.py`:

1. `split_mid` dan `split_bound` sekarang memakai `use_cache=True` + membawa
   `caches` — sehingga attention melihat history penuh, dan check benar-benar
   menguji chunk-invariance full model (konsisten dengan `cache_split` yang
   memang sudah OK).
2. Ditambah check baru **SSM-only split (nocache)** yang menguji jalur
   chunk-invariant yang sebenarnya (cabang SSM) langsung di level
   `ChunkwiseSSM`, tanpa cache — ini yang dijamin invariant.

## Catatan desain (sesuai keterangan penulis)

- Isi slot tetap `state.detach()` (truncated BPTT, tidak ada credit ke masa
  depan lewat memory) — ini desain, bukan bug wiring.
- Loss 230.9 adalah nilai CE model acak belum terlatih (bukan NaN), wajar.
