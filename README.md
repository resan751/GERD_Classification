# Panduan Sistem G-Care 
Sistem Klasifikasi Citra Endoskopi Berbasis Deep Learning (EfficientNetB5), Image-Type Gate (EfficientNetB4), dan Explainable AI (Grad-CAM)

---

## 1. Ikhtisar Sistem & Arsitektur

**GERDGuard** adalah sistem pendukung keputusan klinis (*Clinical Decision Support System* / CDSS) berbasis kecerdasan buatan (*Deep Learning*) yang dirancang untuk menganalisis citra endoskopi saluran cerna atas (*Esophagogastroduodenoscopy* / EGD). Sistem ini membantu menyaring tanda-tanda erosi mukosa esofagus distal dan kelainan pada esophagogastric junction (Z-line) yang berkaitan dengan GERD (*Gastroesophageal Reflux Disease*).

Sistem mengintegrasikan pipeline tiga tahap yang aman, transparan, dan dapat dipertanggungjawabkan:
1. **Tahap 1 — Image-Type Gate Guard (EfficientNetB4, 380×380):** Menyaring dan memverifikasi integritas citra masukan untuk menolak citra non-endoskopi (foto luar, teks, dokumen, radiologi).
2. **Tahap 2 — GERD Classifier (EfficientNetB5, 456×456):** Mengklasifikasikan citra ke dalam dua kategori diagnostik:
   - **`GERD`**: Esofagitis Erosif (terdapat mucosal breaks / lesi refluks esofagus).
   - **`GERD_Normal`**: Mukosa esofagus dalam batas normal atau Non-Erosive Reflux Disease (NERD).
3. **Tahap 3 — Explainable AI Grad-CAM & Patient Report Generator :** Mengomputasi peta perhatian gradien (*Class Activation Mapping*) pada layer `top_conv` untuk transparansi visual inferensi AI, serta menyusun laporan diagnostik individual komprehensif.

```
[Citra Unggahan Pasien]
          │
          ▼
┌─────────────────────────────────────────┐
│ 1. Image-Type Gate (EfficientNetB4)     │ ──▶ Ditolak (Bukan Endoskopi / 422)
│    Input: RGB 380×380                   │
└─────────────────────────────────────────┘
          │ (LOLOS_GATE)
          ▼
┌─────────────────────────────────────────┐
│ 2. GERD Classifier (EfficientNetB5)     │ ──▶ Probabilitas Sigmoid P(GERD)
│    Input: RGB 456×456                   │
└─────────────────────────────────────────┘
          │
          ├───────────────────────────────┐
          ▼                               ▼
┌───────────────────────────┐   ┌────────────────────────────────────────┐
│ 3. Grad-CAM Explainer     │   │ 4. Patient Report Generator v2.0       │
│    Layer: top_conv        │   │    - Ringkasan Narasi Klinis AI        │
│    Resolusi: 15 × 15 × 2048│  │    - Tingkat Keyakinan & Gate          │
│    Adaptive Alpha Overlay │   │    - Distribusi 9 Area Grid            │
└───────────────────────────┘   │    - Tabel Temuan Klinis (Belum Teredia)│
          │                     │    - Catatan Klinis & Disclaimer       │
          └───────────────┬─────┴────────────────────────────────────────┘
                          ▼
             [Medical AI Report Dashboard]
```

---

## 2. Stack Teknologi yang Digunakan

### Backend (`/backend`)
| Komponen | Teknologi | Peran |
|---|---|---|
| **Runtime** | Python 3.10+ / 3.11 | Lingkungan eksekusi backend |
| **Web Framework** | FastAPI | Framework REST API asinkron berkinerja tinggi |
| **ASGI Server** | Uvicorn | Server HTTP/ASGI produksi dan pengembangan |
| **Deep Learning** | TensorFlow / Keras | Pemuatan model `.keras` dan eksekusi inferensi |
| **Model Klasifikasi** | EfficientNetB5 (`model/gerd/model_gerd_efficientnetb5.keras`) | Model transfer learning resolusi tinggi 456×456 |
| **Image-Type Gate** | EfficientNetB4 (`model/gate/image_type_classifier_efficientnetb4.keras`) | Filter keamanan penyaring citra non-endoskopi 380×380 |
| **Explainable AI** | Grad-CAM (Gradient-weighted Class Activation Mapping) | Ekstraksi aktivasi gradien layer `top_conv` (15×15×2048) |
| **Report Generator** | Python Engine v2.0 (`report_generator.py`) | Mesin sintesis laporan klinis dan kuantisasi 9 region heatmap |
| **Image Processing** | OpenCV (cv2) & Pillow (PIL) | Manipulasi matriks BGR/RGB, colormap JET, adaptive alpha blending, dan encoding Base64 |
| **Komputasi Matriks** | NumPy | Operasi tensor, clipping persentil, dan normalisasi |
| **Upload Parser** | python-multipart | Penanganan pengunggahan berkas multipart/form-data |

### Frontend (`/frontend`)
| Komponen | Teknologi | Peran |
|---|---|---|
| **Framework Web** | Next.js 15 (App Router) | Framework React modern untuk performa tinggi & SSR |
| **UI Library** | React 19 | Library antarmuka komponen deklaratif |
| **Bahasa** | TypeScript 5.6 | Menjamin keamanan tipe (*type safety*) kode |
| **Desain & Tema** | CSS Variables Modern (`globals.css`) | Tema dinamis Terang (*Light Mode*) dan Gelap (*Dark Mode*) |
| **Medical Report View** | `AnalysisReportDashboard.tsx` | Dashboard visualisasi klinis terstruktur (Section A - G) |
| **Modal Riwayat** | `ResultDialog.tsx` | Widescreen medical dialog responsif ramah cetak (`@media print`) |
| **Penyimpanan Lokal** | Browser LocalStorage (`lib/store.ts`) | Penyimpanan riwayat pemeriksaan & preferensi ambang batas |
| **Proxy Gateway** | Next.js Rewrites (`next.config.mjs`) | Meneruskan `/api/*` ke backend tanpa kendala CORS |

---

## 3. Prasyarat Sistem (*Prerequisites*)

Sebelum memulai instalasi, pastikan perangkat Anda telah terpasang:
1. **Python 3.11 (64-bit)** ([Unduh Python](https://www.python.org/downloads/))
2. **Node.js (v18.18+ atau v20+) & npm** ([Unduh Node.js](https://nodejs.org/))
3. **Git** (untuk manajemen repositori)
4. **Sistem Operasi**: Windows 10/11, macOS, atau Linux

---

## 4. Panduan Instalasi & Penyiapan Pertama Kali

### Langkah A: Penyiapan Backend
1. Buka terminal (PowerShell / Command Prompt / Terminal) dan masuk ke direktori `backend`:
   ```powershell
   cd backend
   ```

2. Buat dan aktifkan virtual environment Python:
   ```powershell
   # Membuat virtual environment
   python -m venv .venv

   # Mengaktifkan di Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Atau jika di Command Prompt (cmd):
   .\.venv\Scripts\activate.bat
   # Atau jika di macOS / Linux:
   source .venv/bin/activate
   ```

3. Pasang seluruh dependensi backend:
   ```powershell
   pip install -r requirements.txt
   ```

4. Pastikan berkas model dan konfigurasi berikut berada pada direktori `backend/model/`:
   - `model/gerd/model_gerd_efficientnetb5.keras` *(Model klasifikasi aktif EfficientNetB5)*
   - `model/gate/image_type_classifier_efficientnetb4.keras` *(Model Image-Type Gate EfficientNetB4)*
   - `model/gate/image_type_config.json` *(Konfigurasi ambang verifikasi non-endoskopi)*

---

### Langkah B: Penyiapan Frontend
1. Buka terminal baru dan masuk ke folder `frontend`:
   ```powershell
   cd frontend
   ```

2. Pasang dependensi Node.js:
   ```powershell
   npm install
   ```

---

## 5. Cara Menjalankan Aplikasi (*Running the Application*)

### Cara Cepat: 1 Kali Klik (*One-Click Launcher*)
Anda dapat menjalankan Backend FastAPI dan Frontend Next.js secara bersamaan dengan **1 kali klik**:
- **Metode A (Double-Click di Windows Explorer):** Cukup klik dua kali berkas **`run-all.bat`**.
- **Metode B (Via PowerShell):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\run-all.ps1
  ```
*Skrip ini secara otomatis memvalidasi lingkungan virtual `.venv`, memasang `node_modules` jika belum ada, membuka terminal backend (port 8000) dan frontend (port 3000), serta membuka peramban secara otomatis ke `http://localhost:3000`.*

>  **Untuk Menghentikan Seluruh Layanan:** Klik dua kali berkas **`stop-all.bat`** atau jalankan `.\stop-all.ps1`.

---

### Cara Manual (Dua Terminal Terpisah):

#### Terminal 1: Jalankan Backend API (FastAPI)
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
*Tanda backend berhasil aktif:*
```text
========== GERDGuard STARTUP ==========
[1/2] Loading EfficientNetB5...
[2/2] Loading Image Type Gate...

GERDGuard API siap.
Model GERD      : ...\backend\model\gerd\model_gerd_efficientnetb5.keras
Image-type gate : ...\backend\model\gate\image_type_classifier_efficientnetb4.keras
Gate threshold  : 0.01
GERD threshold  : 0.5
=======================================
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

#### Terminal 2: Jalankan Frontend Web (Next.js)
```powershell
cd frontend
npm run dev
```
*Tanda frontend berhasil aktif:*
```text
   ▲ Next.js 15.1.0 (atau 15.5+)
   - Local:        http://localhost:3000
```

### Akses Melalui Peramban (*Browser*):
-  **Antarmuka Aplikasi Web**: [http://localhost:3000](http://localhost:3000)
-  **Dokumentasi Interaktif API (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
-  **Pemeriksaan Kesiapan Model (*Health Check*)**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 6. Panduan Penggunaan Modul Aplikasi

### 1. Halaman Beranda (`/`)
- Memuat ringkasan kapabilitas sistem, informasi arsitektur **EfficientNetB5**, sistem keamanan **Image-Type Gate**, alur kerja pemeriksaan, serta tombol cepat **"Mulai Analisis"**.

### 2. Modul Analisis (`/analisis`)
- **Informasi Pasien (Grid 2 Kolom):** Masukkan ID Pasien Anonim (misal: `PS-8821`), Usia, Jenis Kelamin, dan Catatan Keluhan/Indikasi Klinis.
- **Unggah Citra Endoskopi:**
  - Area *dropzone* responsif dengan *feedback* visual *drag-and-drop*.
  - Format yang didukung: **JPG, JPEG, PNG, BMP** (maksimal 10 MB).
  - Pratinjau citra proporsional (*object-fit: contain*) tanpa terpotong, dilengkapi nama berkas, ukuran, tombol **Ganti Berkas**, dan **Hapus**.
- **Indikator Proses Jujur (*Honest Pipeline Stepper*):**
  Saat tombol analisis ditekan, sistem menampilkan indikator tahapan nyata tanpa memalsukan persentase:
  1. Validasi format & integritas piksel gambar
  2. Pemeriksaan Image-Type Gate (380×380)
  3. Inferensi klasifikasi EfficientNetB5 (456×456)
  4. Komputasi peta perhatian Grad-CAM (top_conv)
  5. Penyusunan laporan klinis individual (Report Generator v2.0)

### 3. Riwayat Pemeriksaan (`/riwayat`)
- Menyimpan rekam medis analisis secara lokal di browser (*LocalStorage*).
- Dilengkapi pencarian ID Pasien, filter cepat (*Semua*, *GERD*, *Normal*), pagination, fitur seleksi baris (*checklist/voting*), tombol **Hapus yang Dipilih**, **Hapus Semua**, dan **Ekspor CSV**.
- Klik tombol **Detail** pada baris tabel akan membuka **Widescreen Medical Dialog** (`ResultDialog.tsx`) yang menyajikan laporan diagnostik lengkap siap cetak.

### 4. Pengaturan Profil (`/profil`)
- Mengatur identitas pengguna, institusi medis, dan peran klinis.
- Mengatur **Ambang Batas Keyakinan (*Threshold*)** kustom ($50\% - 98\%$) dengan rekomendasi standar $85\%$.

---

## 7. Pengujian Model di RapidMiner Studio

Untuk mengevaluasi performa model (*Confusion Matrix, Kurva ROC, AUC, F1-Score, Presisi, Sensitivitas*) menggunakan **RapidMiner Studio**:

1. Jalankan skrip evaluasi batch bawaan:
   ```powershell
   cd backend
   python export_for_rapidminer.py --folder "testing_guard" --model "model/gerd/model_gerd_efficientnetb5.keras" --out "hasil_prediksi_rapidminer.csv"
   ```
2. Buka **RapidMiner Studio** $\rightarrow$ Buat proses baru:
   - Tarik operator **`Read CSV`** (arahkan ke berkas `hasil_prediksi_rapidminer.csv`).
   - Tarik operator **`Set Role`** (pilih atribut `actual_label` $\rightarrow$ set target role ke `label`).
   - Tarik operator **`Set Role`** (pilih atribut `prediction` $\rightarrow$ set target role ke `prediction`).
   - Tarik operator **`Performance (Binominal Classification)`** $\rightarrow$ aktifkan *accuracy*, *precision*, *recall*, *AUC*, dan *f_measure*.
3. Hubungkan port operator dan klik tombol **Run** untuk melihat matriks konfusi visual dan kurva performa.

---

## 8. Spesifikasi Endpoint API Backend (FastAPI)

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Informasi status API, versi, tautan dokumentasi Swagger, dan health check. |
| `GET` | `/health` | Memeriksa kesiapan model EfficientNetB5, Image-Type Gate, threshold, dan status server. |
| `POST` | `/predict` | Menerima berkas citra `multipart/form-data` (`file`) dan menghasilkan inferensi klasifikasi, Grad-CAM, serta laporan v2.0. |

### Struktur Respons Sukses Citra Lolos Gate (`200 OK`):
```json
{
  "status": "GERD",
  "prediction": "GERD",
  "predicted_class": "GERD",
  "label": "GERD",
  "probability_gerd": 0.941258,
  "probability_normal": 0.058742,
  "confidence": 0.941258,
  "gate_status": "LOLOS_GATE",
  "gate_score": 0.000124,
  "model_name": "EfficientNetB5",
  "original_image": "data:image/jpeg;base64,...",
  "heatmap_image": "data:image/jpeg;base64,...",
  "overlay_image": "data:image/jpeg;base64,...",
  "report": {
    "report_version": "2.0",
    "classification": {
      "predicted_class": "GERD",
      "p_gerd": 0.941258,
      "p_normal": 0.058742,
      "confidence": 0.941258,
      "confidence_percentage": 94.13,
      "confidence_level": "Tinggi",
      "confidence_description": "Model menunjukkan nilai probabilitas yang tinggi terhadap kelas prediksi.",
      "explanation": "EfficientNetB5 mengklasifikasikan citra sebagai GERD.",
      "detail": "Probabilitas GERD sebesar 94.13%, sedangkan probabilitas GERD_Normal sebesar 5.87%."
    },
    "image_gate": {
      "status": "LOLOS_GATE",
      "explanation": "Citra berhasil melewati pemeriksaan Image Type Gate."
    },
    "visualization": {
      "available": true,
      "dominant_region": "tengah citra",
      "dominant_activation_percentage": 38.45,
      "activation_pattern": "localized",
      "activation_pattern_description": "Aktivasi relatif terkonsentrasi pada satu area.",
      "coverage_percentage": 24.18,
      "region_distribution": [
        { "region": "tengah citra", "score": 0.452, "percentage": 38.45 },
        { "region": "kiri tengah", "score": 0.210, "percentage": 17.86 },
        { "region": "kanan tengah", "score": 0.165, "percentage": 14.03 }
      ],
      "top_regions": [ ... ],
      "explanation": "Aktivasi tertinggi berada di bagian tengah citra dengan distribusi 38.45%..."
    },
    "findings": {
      "lesion_detection": "not_available",
      "erosion_detection": "not_available",
      "inflammation_detection": "not_available",
      "ulcer_detection": "not_available",
      "la_grade": "not_available",
      "explanation": "Model klasifikasi saat ini belum dilatih untuk mendeteksi erosi, inflamasi, ulserasi, atau menentukan LA Grade."
    },
    "explanation": {
      "summary": "EfficientNetB5 mengklasifikasikan citra sebagai GERD...",
      "prediction_interpretation": "EfficientNetB5 mengklasifikasikan citra sebagai GERD.",
      "probability_interpretation": "Probabilitas GERD sebesar 94.13%...",
      "visual_interpretation": "Model menunjukkan aktivasi tertinggi pada area tengah citra...",
      "clinical_note": "Hasil ini merupakan keluaran model klasifikasi berbasis citra. Grad-CAM menunjukkan area visual yang berkontribusi terhadap prediksi model..."
    }
  }
}
```

### Struktur Respons Citra Non-Endoskopi Ditolak Gate (`200 OK` / Penanganan Gate):
```json
{
  "status": "DITOLAK",
  "prediction": "DITOLAK",
  "predicted_class": null,
  "label": null,
  "probability_gerd": null,
  "probability_normal": null,
  "confidence": null,
  "gate_status": "DITOLAK",
  "gate_score": 0.998412,
  "model_name": "EfficientNetB5",
  "reason": "Gambar terdeteksi sebagai non-endoskopi oleh Image Type Gate.",
  "original_image": null,
  "heatmap_image": null,
  "overlay_image": null,
  "report": null
}
```

---

## 9. Verifikasi Kualitas & Pengujian Sistem

Sistem telah diuji menggunakan rangkaian penjaminan mutu otomatis:
1. **Frontend TypeScript Type Check**:
   ```powershell
   cd frontend
   node ./node_modules/typescript/bin/tsc --noEmit
   # Exit Code 0 (Bebas kesalahan tipe data)
   ```
2. **Next.js Production Build**:
   ```powershell
   cd frontend
   npm run build
   # ✓ Compiled successfully
   # ✓ Generating static pages (11/11)
   ```
3. **Pengujian Integrasi Komprehensif Frontend-Backend**:
   ```powershell
   cd frontend
   node test-frontend-integration.mjs
   # Seluruh 9 skenario integrasi berhasil (Health check, upload FormData, lolos gate, tolak gate, validasi base64 Grad-CAM, validasi report v2.0, fallback port offline).
   ```

---

## 10. Panduan Mengatasi Kendala (*Troubleshooting*)

1. **PowerShell Script Execution Error (`PSSecurityException`):**
   Jika aktivasi virtual environment diblokir oleh sistem Windows PowerShell, jalankan perintah ini satu kali:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```
2. **Port 8000 atau Port 3000 Terpakai (*Port already in use*):**
   ```powershell
   # Cari PID yang menggunakan port:
   netstat -ano | findstr :8000
   # Hentikan proses bersangkutan:
   taskkill /PID <NOMOR_PID> /F
   ```
3. **Backend Tidak Terjangkau dari Frontend:**
   - Pastikan backend aktif di `http://127.0.0.1:8000`.
   - Frontend Next.js secara otomatis mem-proxy panggilan `/api/*` ke `http://127.0.0.1:8000/*` melalui konfigurasi `next.config.mjs`.
4. **Format Berkas Tidak Didukung:**
   - Gunakan format **JPG**, **JPEG**, **PNG**, atau **BMP**. Berkas berformat lain atau citra yang rusak akan ditolak oleh parser validasi sebelum diproses lebih lanjut.

---

## 11. Batasan Tanggung Jawab Klinis (*Medical Disclaimer*)

>  **PERINGATAN MEDIS PENTING:**
> Sistem **GERDGuard** dikembangkan sebagai instrumen komputasi riset dan pendukung keputusan klinis (*Clinical Decision Support Tool*). Sistem ini **berfungsi menyaring karakteristik pola citra medis dan bukan menetapkan diagnosis medis final secara mandiri**.
> Peta perhatian Grad-CAM menunjukkan fitur visual matematis yang memengaruhi keputusan model dan tidak dapat dianggap sebagai bukti langsung adanya lesi patologis, inflamasi, atau kelainan anatomis definitif. Segala penegakan diagnosis klinis, penetapan derajat Los Angeles (LA Grade), serta keputusan intervensi terapi tetap menjadi hak, wewenang, dan tanggung jawab penuh dokter spesialis gastroenterologi / tenaga medis berwenang.
