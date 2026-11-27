# GERD AI Frontend (Next.js 15)

Antarmuka web modern untuk sistem klasifikasi citra endoskopi GERD berbasis **Next.js 15 App Router**, **React 19**, dan **TypeScript**.

---

## 1. Stack Teknologi Frontend

- **Framework**: Next.js 15 (App Router)
- **UI Library**: React 19
- **Bahasa**: TypeScript 5.6
- **Styling**: CSS Variables terpusat (`app/globals.css`) dengan dukungan Dark Mode otomatis/manual
- **Penyimpanan Data**: Browser LocalStorage (`lib/store.ts`)
- **Proxy Backend**: Next.js Route Rewrites (`next.config.mjs`) untuk meneruskan `/api/*` langsung ke FastAPI (`http://127.0.0.1:8000`)

---

## 2. Cara Menjalankan

Masuk ke folder `frontend`:
```bash
cd frontend

# Salin konfigurasi environment jika diperlukan (opsional, default ke http://127.0.0.1:8000)
cp .env.example .env.local

# Instal dependensi
npm install

# Jalankan server pengembangan
npm run dev
```

Buka peramban di [http://localhost:3000](http://localhost:3000).

---

## 3. Struktur Halaman & Rute

| Rute | Halaman / Modul | Deskripsi |
|---|---|---|
| `/` | Beranda | Informasi sistem, ringkasan model, dan alur kerja |
| `/analisis` | Modul Analisis | Unggah citra endoskopi, tombol sampel uji cepat, formulir pasien anonim, dan modal hasil |
| `/riwayat` | Arsip Pemeriksaan | Riwayat pemeriksaan lokal, filter hasil, pagination, pencarian ID, dan ekspor CSV |
| `/profil` | Profil Pengguna | Pengaturan data klinisi dan kustomisasi ambang batas keyakinan (threshold 50%–98%) |
| `/login` | Masuk | Halaman login klinisi (antarmuka) |
| `/register` | Registrasi | Pendaftaran akun baru |
| `/forgot-password` | Lupa Sandi | Permintaan reset sandi |

---

## 4. Integrasi dengan Backend FastAPI

- **Panggilan API**: Fungsi `predict()` pada `lib/api.ts` memanggil `/api/predict` via method `POST` dengan format payload `multipart/form-data` (`file`).
- **Penolakan Guard**: Jika backend menolak citra non-endoskopi dengan status **HTTP 422**, antarmuka menangkap `ApiError` dan menampilkan kartu peringatan khusus di bawah input citra.
- **Penyimpanan Lokal**: Rekaman hasil analisis disimpan secara lokal di browser melalui kunci `gerd.history` dan profil di `gerd.profile`.
