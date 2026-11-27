"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import AnalysisReportDashboard from "@/components/AnalysisReportDashboard";
import { ApiError, predict, type PredictionResult } from "@/lib/api";
import { addRecord, DEFAULT_PROFILE, makeThumb, PROFILE_KEY, useStored, type HistoryRecord } from "@/lib/store";

const OK_TYPES = ["image/jpeg", "image/png", "image/bmp"];
const OK_EXTS = [".jpg", ".jpeg", ".png", ".bmp"];
const ACCEPT_ATTR = "image/jpeg,image/png,image/bmp,.jpg,.jpeg,.png,.bmp";
const MAX_MB = 10;

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isValidImage(f: File) {
  const typeMatch = OK_TYPES.includes(f.type.toLowerCase());
  const nameMatch = OK_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext));
  return typeMatch || nameMatch;
}

export default function Analisis() {
  const [profile] = useStored(PROFILE_KEY, DEFAULT_PROFILE);
  const input = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [rejected, setRejected] = useState("");
  const [record, setRecord] = useState<HistoryRecord | null>(null);
  const [saved, setSaved] = useState(false);

  // Form patient fields
  const [patientId, setPatientId] = useState("");
  const [age, setAge] = useState<number | "">("");
  const [sex, setSex] = useState("Laki-laki");
  const [notes, setNotes] = useState("");

  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [url]);

  function pick(f?: File) {
    if (!f) return;
    setError("");
    setRejected("");
    setSaved(false);
    setRecord(null);

    if (!isValidImage(f)) {
      return setError("Format berkas harus JPG, JPEG, PNG, atau BMP.");
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      return setError(`Ukuran berkas (${formatSize(f.size)}) melebihi batas maksimal ${MAX_MB} MB.`);
    }

    if (url) URL.revokeObjectURL(url);
    setFile(f);
    setUrl(URL.createObjectURL(f));
  }

  function handleReset() {
    if (url) URL.revokeObjectURL(url);
    setFile(null);
    setUrl("");
    setError("");
    setRejected("");
    setRecord(null);
    setSaved(false);
    if (input.current) input.current.value = "";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!file) return setError("Pilih citra endoskopi terlebih dahulu sebelum memulai analisis.");

    setBusy(true);
    setError("");
    setRejected("");
    setSaved(false);
    setRecord(null);

    try {
      // 1. Kirim file via FormData ke POST /predict backend
      const [predictionRes, thumb] = await Promise.all([predict(file), makeThumb(file)]);

      // Siapkan history record lengkap dengan report v2.0
      const rec: HistoryRecord = {
        id: crypto.randomUUID(),
        patientId: patientId || "Anonim",
        age: typeof age === "number" ? age : 0,
        sex,
        notes,
        at: new Date().toISOString(),
        file: file.name,
        thumb,
        label: predictionRes.label,
        confidence: predictionRes.confidence,
        probabilities: predictionRes.probabilities,
        gate_status: predictionRes.gate_status,
        original_image: predictionRes.original_image,
        heatmap_image: predictionRes.heatmap_image,
        overlay_image: predictionRes.overlay_image,
        model_name: predictionRes.model_name,
        report: predictionRes.report,
      };
      setRecord(rec);

      // Auto scroll ke hasil analisis
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 150);
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 422) {
          // Ditolak oleh Image Type Gate
          setRejected(err.message);
        } else if (err.status === 0) {
          // Backend tidak terhubung
          setError("Backend tidak tersedia. Pastikan server FastAPI aktif di port 8000.");
        } else {
          setError(err.message);
        }
      } else {
        setError(err instanceof Error ? err.message : "Terjadi kesalahan saat analisis citra.");
      }
    } finally {
      setBusy(false);
    }
  }

  function handleSaveHistory() {
    if (record) {
      addRecord(record);
      setSaved(true);
    }
  }

  return (
    <div className="analysis-container">
      {/* =================================================================== */}
      {/* 1. HEADER HALAMAN                                                   */}
      {/* =================================================================== */}
      <nav className="breadcrumb-nav no-print" aria-label="Breadcrumb">
        <Link href="/">Beranda</Link>
        <span>/</span>
        <span style={{ color: "var(--ink)", fontWeight: 600 }}>Analisis Endoskopi</span>
      </nav>

      <div className="analysis-header-box">
        <span className="eyebrow">Medical AI Diagnostic</span>
        <h1 style={{ fontSize: "clamp(28px, 4vw, 42px)", marginTop: 8 }}>
          Analisis Endoskopi
        </h1>
        <p className="muted" style={{ marginTop: 10, maxWidth: "72ch", fontSize: 15, lineHeight: 1.6 }}>
          Unggah citra endoskopi esofagus untuk evaluasi indikasi GERD. Sistem secara otomatis memvalidasi kesesuaian citra melalui <b>Image Type Gate (380×380)</b>, mengklasifikasi dengan <b>EfficientNetB5 (456×456)</b>, serta menyajikan visualisasi <b>Grad-CAM</b> dan laporan klinis terstruktur.
        </p>
      </div>

      {/* =================================================================== */}
      {/* FORM PEMERIKSAAN & UPLOAD                                           */}
      {/* =================================================================== */}
      <form onSubmit={submit} className="stack" style={{ ["--gap" as string]: "28px", marginTop: 36 }}>
        {/* 2. INFORMASI PEMERIKSAAN PASIEN */}
        <section className="medical-card">
          <div className="medical-card-header">
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 700 }}>1. Informasi Pemeriksaan Pasien</h2>
              <p className="small muted" style={{ marginTop: 2 }}>
                Lengkapi identitas dan indikasi klinis untuk dokumentasi laporan.
              </p>
            </div>
            <span className="small meta-pill">Data Privasi Lokal</span>
          </div>

          <div className="form-grid-2">
            {/* ID Pasien */}
            <div className="field">
              <label htmlFor="patientId">ID Pasien</label>
              <input
                id="patientId"
                name="patientId"
                className="input"
                required
                placeholder="Misal: PS-8821 atau MED-042"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
              />
              <span className="hint">Gunakan kode untuk menjaga privasi rekam medis.</span>
            </div>

            {/* Usia Pasien */}
            <div className="field">
              <label htmlFor="age">Usia (Tahun)</label>
              <input
                id="age"
                name="age"
                type="number"
                min={1}
                max={120}
                className="input"
                required
                placeholder="Misal: 45"
                value={age}
                onChange={(e) => setAge(e.target.value === "" ? "" : Number(e.target.value))}
              />
              <span className="hint">Masukkan rentang usia pasien (1 - 120 tahun).</span>
            </div>

            {/* Jenis Kelamin */}
            <div className="field">
              <span className="label">Jenis Kelamin *</span>
              <div className="seg" role="radiogroup" aria-label="Jenis kelamin">
                {["Laki-laki", "Perempuan", "Lainnya"].map((s) => (
                  <label key={s}>
                    <input
                      type="radio"
                      name="sex"
                      value={s}
                      checked={sex === s}
                      onChange={() => setSex(s)}
                    />
                    {s}
                  </label>
                ))}
              </div>
            </div>

            {/* Catatan Keluhan Klinis */}
            <div className="field">
              <label htmlFor="notes">Catatan Indikasi Klinis / Keluhan (Opsional)</label>
              <textarea
                id="notes"
                name="notes"
                className="textarea"
                placeholder="Misal: Heartburn persisten, regurgitasi asam, riwayat esofagitis, disfagia..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
              />
            </div>
          </div>
        </section>

        {/* 3 & 4. UPLOAD & PREVIEW GAMBAR ENDOSKOPI */}
        <section className="medical-card">
          <div className="medical-card-header">
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 700 }}>2. Unggah & Pratinjau Citra Endoskopi</h2>
              <p className="small muted" style={{ marginTop: 2 }}>
                Pilih foto endoskopi esofagus berformat standar medis.
              </p>
            </div>
            <span className="small muted">JPG, JPEG, PNG, BMP · Maks 10 MB</span>
          </div>

          {/* Area Preview atau Dropzone */}
          {url ? (
            /* 4. PREVIEW GAMBAR */
            <div className="preview-card">
              <div className="preview-canvas">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={url}
                  alt="Pratinjau citra endoskopi yang dipilih"
                  className="preview-img"
                />
              </div>

              <div className="preview-footer">
                <div style={{ minWidth: 0, maxWidth: "70%" }}>
                  <div className="small mono" style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {file?.name}
                  </div>
                  <div className="small muted" style={{ marginTop: 2 }}>
                    {file ? formatSize(file.size) : ""} · <span style={{ color: "var(--ok)", fontWeight: 600 }}>Siap Dianalisis</span>
                  </div>
                </div>

                <div className="row" style={{ gap: 8 }}>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ padding: "6px 14px", minHeight: 36, fontSize: 13 }}
                    onClick={() => {
                      if (input.current) input.current.value = "";
                      input.current?.click();
                    }}
                  >
                    Ganti Berkas
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ padding: "6px 14px", minHeight: 36, fontSize: 13, color: "var(--danger)" }}
                    onClick={handleReset}
                  >
                    Hapus
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* 3. DROPZONE UPLOAD */
            <div
              role="button"
              tabIndex={0}
              className={`dropzone-container ${over ? "over" : ""}`}
              onClick={() => {
                if (input.current) input.current.value = "";
                input.current?.click();
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  if (input.current) input.current.value = "";
                  input.current?.click();
                }
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setOver(true);
              }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(false);
                pick(e.dataTransfer.files?.[0]);
              }}
            >
              <div className="dropzone-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>

              <div>
                <p style={{ fontWeight: 600, fontSize: 16 }}>
                  Tarik & letakkan citra endoskopi ke sini, atau klik untuk memilih
                </p>
                <p className="hint" style={{ marginTop: 6 }}>
                  Mendukung format JPG, JPEG, PNG, atau BMP · Ukuran maksimal 10 MB
                </p>
              </div>

              <span className="badge" style={{ background: "var(--surface-2)", marginTop: 4 }}>
                Pemeriksaan Otomatis oleh Image Type Gate
              </span>
            </div>
          )}

          <input
            ref={input}
            id="image-file-input"
            type="file"
            accept={ACCEPT_ATTR}
            hidden
            onChange={(e) => {
              pick(e.target.files?.[0]);
              e.target.value = "";
            }}
            aria-label="Pilih citra endoskopi"
          />

          {/* Alert Penolakan Image Gate */}
          {rejected && (
            <div className="notice bad" role="alert" style={{ marginTop: 16, display: "flex", gap: 12, alignItems: "flex-start" }}>
              <span style={{ fontSize: 24, lineHeight: 1 }}>🛡️</span>
              <div>
                <b>Citra Ditolak oleh Image Type Gate:</b>
                <p style={{ marginTop: 4, lineHeight: 1.5 }}>{rejected}</p>
                <p className="small muted" style={{ marginTop: 6 }}>
                  Sistem keamanan Image Type Gate mendeteksi bahwa berkas bukan citra endoskopi esofagus/lambung medis. Model inferensi EfficientNetB5 dan visualisasi Grad-CAM tidak dijalankan untuk menjaga integritas diagnostik.
                </p>
              </div>
            </div>
          )}

          {/* Alert Error Umum / Backend Offline */}
          {error && (
            <div className="notice bad" role="alert" style={{ marginTop: 16, display: "flex", gap: 12, alignItems: "flex-start" }}>
              <span style={{ fontSize: 24, lineHeight: 1 }}></span>
              <div>
                <b>Gagal Memproses Analisis:</b>
                <p style={{ marginTop: 4 }}>{error}</p>
              </div>
            </div>
          )}

          {/* 5. TOMBOL ANALISIS (SETELAH PREVIEW) */}
          <div style={{ marginTop: 24, display: "flex", flexDirection: "column", gap: 12 }}>
            <button
              type="submit"
              className="btn btn-primary btn-mobile-full"
              style={{ padding: "14px 28px", fontSize: 16, fontWeight: 700 }}
              disabled={busy || !file}
            >
              {busy ? (
                <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
                  <span className="pipeline-spinner" />
                  Menganalisis Citra & Grad-CAM…
                </span>
              ) : (
                "Mulai Analisis Endoskopi"
              )}
            </button>

            {!file && (
              <p className="hint" style={{ textAlign: "center" }}>
                * Pilih atau tarik citra endoskopi terlebih dahulu untuk mengaktifkan tombol analisis.
              </p>
            )}
          </div>

          {/* 6. LOADING STATE (HONEST PROCESS PIPELINE) */}
          {busy && (
            <div className="loading-pipeline-box" role="status" aria-live="polite">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span style={{ fontWeight: 700, fontSize: 14, color: "var(--primary)" }}>
                  Tahapan Pemrosesan Model AI GERDGuard:
                </span>
                <span className="badge" style={{ background: "var(--surface-2)" }}>Sedang Berjalan…</span>
              </div>

              <div className="loading-pipeline-list">
                <div className="pipeline-item active">
                  <span className="pipeline-spinner" />
                  <span>1. Memvalidasi format & integritas piksel gambar</span>
                </div>
                <div className="pipeline-item active">
                  <span className="pipeline-spinner" />
                  <span>2. Memeriksa kesesuaian citra via Image Type Gate (380×380)</span>
                </div>
                <div className="pipeline-item active">
                  <span className="pipeline-spinner" />
                  <span>3. Menjalankan inferensi klasifikasi model EfficientNetB5 (456×456)</span>
                </div>
                <div className="pipeline-item active">
                  <span className="pipeline-spinner" />
                  <span>4. Mengomputasi peta keterbukaan aktivasi Grad-CAM (top_conv)</span>
                </div>
                <div className="pipeline-item active">
                  <span className="pipeline-spinner" />
                  <span>5. Menyusun laporan analisis klinis individual (Report Generator v2.0)</span>
                </div>
              </div>
            </div>
          )}
        </section>
      </form>

      {/* =================================================================== */}
      {/* 7. HASIL ANALISIS (MEDICAL AI REPORT DASHBOARD)                     */}
      {/* =================================================================== */}
      {record && (
        <section ref={resultsRef} style={{ marginTop: 52 }}>
          <AnalysisReportDashboard
            record={record}
            threshold={profile.threshold}
            onReset={handleReset}
            onSave={handleSaveHistory}
            saved={saved}
          />
        </section>
      )}
    </div>
  );
}
