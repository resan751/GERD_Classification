"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { HistoryRecord } from "@/lib/store";

export type AnalysisReportDashboardProps = {
  record: HistoryRecord;
  threshold?: number;
  onReset?: () => void;
  onSave?: () => void;
  saved?: boolean;
  isDialog?: boolean;
  onClose?: () => void;
};

type ViewMode = "compare" | "heatmap" | "all";

const pct = (n?: number | null) => (typeof n === "number" ? `${(n * 100).toFixed(1)}%` : "-");

export default function AnalysisReportDashboard({
  record,
  threshold = 85,
  onReset,
  onSave,
  saved = false,
  isDialog = false,
  onClose,
}: AnalysisReportDashboardProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("compare");
  const [activeLightbox, setActiveLightbox] = useState<{ src: string; title: string; desc: string } | null>(null);

  const isGerd = record.label === "GERD";
  const report = record.report;
  const isLowConfidence = record.confidence * 100 < threshold;

  // Format tanggal & waktu
  const formattedDate = record.at
    ? new Date(record.at).toLocaleString("id-ID", {
        day: "numeric",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        timeZoneName: "short",
      })
    : "-";

  // Handle escape key for lightbox
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && activeLightbox) {
        setActiveLightbox(null);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeLightbox]);

  // Sembilan region default jika data report belum ada
  const defaultRegions = [
    { region: "kiri atas", percentage: 0 },
    { region: "tengah atas", percentage: 0 },
    { region: "kanan atas", percentage: 0 },
    { region: "kiri tengah", percentage: 0 },
    { region: "tengah citra", percentage: 0 },
    { region: "kanan tengah", percentage: 0 },
    { region: "kiri bawah", percentage: 0 },
    { region: "tengah bawah", percentage: 0 },
    { region: "kanan bawah", percentage: 0 },
  ];

  const regionList = report?.visualization?.region_distribution?.length
    ? report.visualization.region_distribution
    : defaultRegions;

  const dominantRegion = report?.visualization?.dominant_region || null;

  return (
    <div className="report-dashboard-container">
      {/* =================================================================== */}
      {/* SECTION A: REPORT HEADER                                            */}
      {/* =================================================================== */}
      <header className="report-header-panel">
        <div className="report-header-top">
          <div>
            <div className="report-badge-row">
              <span className={`status-pill ${isGerd ? "warn" : "ok"}`}>
                ● Status Analisis Selesai
              </span>
              <span className="meta-pill">
                Gate: {record.gate_status || "LOLOS_GATE"}
              </span>
              <span className="meta-pill">
                Model: {record.model_name || "EfficientNetB5"}
              </span>
              {report?.report_version && (
                <span className="meta-pill">v{report.report_version}</span>
              )}
            </div>
            <h1 className="report-main-title">Laporan Hasil Analisis Endoskopi</h1>
            <p className="report-sub-title">
              Evaluasi klasifikasi citra endoskopi esofagus berbasis kecerdasan buatan dan visualisasi keterbukaan Grad-CAM.
            </p>
          </div>

          <div className="report-patient-meta">
            <div className="patient-id-badge">
              <span className="patient-id-label">ID Pasien:</span>
              <span className="patient-id-value mono">{record.patientId || "Anonim"}</span>
            </div>
            <div className="patient-details-text">
              <span>{record.age ? `${record.age} tahun` : "Usia: -"}</span>
              <span>·</span>
              <span>{record.sex || "-"}</span>
            </div>
            <div className="patient-date-text mono">{formattedDate}</div>
          </div>
        </div>

        {/* Toolbar Aksi Header */}
        <div className="report-toolbar no-print">
          <div className="toolbar-left">
            <button
              type="button"
              className="btn btn-ghost toolbar-btn"
              onClick={() => window.print()}
              title="Cetak atau simpan laporan sebagai PDF"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 6 2 18 2 18 9" />
                <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2" />
                <rect x="6" y="14" width="12" height="8" />
              </svg>
              Cetak Laporan
            </button>

            {onSave && !saved && (
              <button
                type="button"
                className="btn btn-primary toolbar-btn"
                onClick={onSave}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                  <polyline points="17 21 17 13 7 13 7 21" />
                  <polyline points="7 3 7 8 15 8" />
                </svg>
                Simpan ke Riwayat
              </button>
            )}

            {saved && (
              <span className="badge ok-soft" style={{ padding: "8px 12px", fontSize: 13 }}>
                ✓ Tersimpan di Riwayat (<Link href="/riwayat">Buka Riwayat</Link>)
              </span>
            )}
          </div>

          <div className="toolbar-right">
            {onReset && (
              <button
                type="button"
                className="btn btn-ghost toolbar-btn"
                onClick={onReset}
              >
                + Analisis Citra Baru
              </button>
            )}

            {isDialog && onClose && (
              <button
                type="button"
                className="btn btn-ghost toolbar-btn"
                onClick={onClose}
              >
                Tutup Dialog
              </button>
            )}
          </div>
        </div>
      </header>

      {/* =================================================================== */}
      {/* SECTION B: CLASSIFICATION SUMMARY                                   */}
      {/* =================================================================== */}
      <section className="report-section">
        <div className="section-title-wrap">
          <span className="eyebrow">Ringkasan Klasifikasi AI</span>
          <h2 className="section-heading">Status Prediksi & Probabilitas Model</h2>
        </div>

        <div className="classification-grid">
          {/* Kartu 1: Prediksi Utama */}
          <div className={`summary-card prediction-card ${isGerd ? "is-gerd" : "is-normal"}`}>
            <div className="card-top-tag">
              <span>Hasil Klasifikasi Model</span>
              <span className={`confidence-badge ${report?.classification?.confidence_level === "Tinggi" ? "high" : "med"}`}>
                Keyakinan {report?.classification?.confidence_level || (record.confidence >= 0.85 ? "Tinggi" : "Sedang")}
              </span>
            </div>

            <div className="prediction-label-lg">
              {isGerd ? "GERD" : "GERD_Normal"}
            </div>

            <p className="prediction-clinical-desc">
              {isGerd
                ? "Citra endoskopi menunjukkan fitur visual yang konsisten dengan Gastroesophageal Reflux Disease (GERD)."
                : "Citra endoskopi berada dalam batas normal tanpa indikasi karakteristik refluks esofagus yang signifikan."}
            </p>

            <div className="prediction-stats-row">
              <div>
                <span className="stats-label">Tingkat Keyakinan</span>
                <div className="stats-value mono">{pct(record.confidence)}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <span className="stats-label">Arsitektur</span>
                <div className="stats-value mono" style={{ fontSize: 16 }}>{record.model_name || "EfficientNetB5"}</div>
              </div>
            </div>

            <div className="confidence-meter-bar">
              <div
                className={`meter-fill ${isGerd ? "warn" : "ok"}`}
                style={{ width: pct(record.confidence) }}
              />
            </div>

            <p className="meter-caption">
              Ambang batas klinis profil Anda: <b>{threshold}%</b>.{" "}
              {isLowConfidence && (
                <span style={{ color: "var(--warn)", fontWeight: 600 }}>
                  (Keyakinan di bawah ambang batas klinis, disarankan tinjauan mendalam)
                </span>
              )}
            </p>
          </div>

          {/* Kartu 2: Distribusi Probabilitas */}
          <div className="summary-card probability-card">
            <div className="card-top-tag">
              <span>Distribusi Probabilitas Kelas</span>
              <span className="meta-pill">Sigmoid Output</span>
            </div>

            <p className="prob-desc-text">
              Proporsi skor probabilitas keluaran neuron klasifikasi model EfficientNetB5:
            </p>

            <div className="prob-bars-stack">
              {/* Probabilitas GERD */}
              <div className="prob-row">
                <div className="prob-header">
                  <span className="prob-name">
                    <span className="color-dot warn" /> P(GERD) — Kelas 1
                  </span>
                  <span className="prob-pct mono warn">{pct(record.probabilities?.GERD)}</span>
                </div>
                <div className="prob-progress-bg">
                  <div
                    className="prob-progress-fill warn"
                    style={{ width: pct(record.probabilities?.GERD) }}
                  />
                </div>
              </div>

              {/* Probabilitas GERD_Normal */}
              <div className="prob-row">
                <div className="prob-header">
                  <span className="prob-name">
                    <span className="color-dot ok" /> P(GERD_Normal) — Kelas 0
                  </span>
                  <span className="prob-pct mono ok">
                    {pct(record.probabilities?.GERD_Normal ?? record.probabilities?.Normal)}
                  </span>
                </div>
                <div className="prob-progress-bg">
                  <div
                    className="prob-progress-fill ok"
                    style={{ width: pct(record.probabilities?.GERD_Normal ?? record.probabilities?.Normal) }}
                  />
                </div>
              </div>
            </div>

            <div className="prob-footer-note">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="16" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12.01" y2="8" />
              </svg>
              <span>
                Nilai probabilitas merefleksikan bobot fitur komputasi matematis model, bukan kepastian diagnosis klinis definitif.
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* =================================================================== */}
      {/* SECTION C: GRAD-CAM VISUALIZATION (AREA LEBIH BESAR & INTERAKTIF)   */}
      {/* =================================================================== */}
      <section className="report-section">
        <div className="section-title-wrap" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
          <div>
            <span className="eyebrow">Visualisasi Keterbukaan Model AI</span>
            <h2 className="section-heading">Peta Aktivasi Grad-CAM (top_conv)</h2>
            <p className="section-subtitle">
              Layer target: <code className="mono">top_conv</code> (EfficientNetB5) · Resolusi fitur: 15 × 15 × 2048
            </p>
          </div>

          {/* Mode Switcher */}
          <div className="view-mode-tabs no-print">
            <button
              type="button"
              className={`mode-tab ${viewMode === "compare" ? "active" : ""}`}
              onClick={() => setViewMode("compare")}
            >
              Bandingan (Asli & Overlay)
            </button>
            <button
              type="button"
              className={`mode-tab ${viewMode === "heatmap" ? "active" : ""}`}
              onClick={() => setViewMode("heatmap")}
            >
              Heatmap JET
            </button>
            <button
              type="button"
              className={`mode-tab ${viewMode === "all" ? "active" : ""}`}
              onClick={() => setViewMode("all")}
            >
              Semua (3 Citra)
            </button>
          </div>
        </div>

        {/* Galeri Gambar Sesuai View Mode */}
        <div className={`gradcam-gallery-container mode-${viewMode}`}>
          {/* 1. Citra Asli */}
          {(viewMode === "compare" || viewMode === "all") && (
            <div className="image-card">
              <div className="image-card-header">
                <span className="image-title">1. Citra Endoskopi Asli</span>
                <span className="badge">Masukan Klinis</span>
              </div>
              <div
                className="image-frame"
                onClick={() =>
                  setActiveLightbox({
                    src: record.original_image || "",
                    title: "Citra Endoskopi Asli",
                    desc: `Berkas: ${record.file} · Resolusi masukan klinis`,
                  })
                }
                title="Klik untuk memperbesar"
              >
                {record.original_image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={record.original_image} alt="Citra Endoskopi Asli" />
                ) : (
                  <div className="no-image-placeholder">Citra asli tidak tersedia</div>
                )}
                <div className="expand-overlay no-print">
                  <span> Klik untuk Perbesar</span>
                </div>
              </div>
              <div className="image-caption">
                Resolusi asli hasil endoskopi pasien sebelum inferensi.
              </div>
            </div>
          )}

          {/* 2. Grad-CAM Overlay */}
          {(viewMode === "compare" || viewMode === "all") && (
            <div className="image-card">
              <div className="image-card-header">
                <span className="image-title" style={{ color: isGerd ? "var(--warn)" : "var(--ok)" }}>
                  2. Grad-CAM Overlay
                </span>
                <span className="badge" style={{ background: "var(--surface-2)" }}>Adaptive Alpha</span>
              </div>
              <div
                className="image-frame"
                onClick={() =>
                  setActiveLightbox({
                    src: record.overlay_image || "",
                    title: "Grad-CAM Overlay",
                    desc: "Superimposisi peta gradien aktivasi pada citra asli",
                  })
                }
                title="Klik untuk memperbesar"
              >
                {record.overlay_image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={record.overlay_image} alt="Grad-CAM Overlay" />
                ) : (
                  <div className="no-image-placeholder">Overlay tidak tersedia</div>
                )}
                <div className="expand-overlay no-print">
                  <span> Klik untuk Perbesar</span>
                </div>
              </div>
              <div className="image-caption">
                Superimposisi visualisasi fokus keputusan model pada struktur esofagus.
              </div>
            </div>
          )}

          {/* 3. Heatmap JET */}
          {(viewMode === "heatmap" || viewMode === "all") && (
            <div className="image-card">
              <div className="image-card-header">
                <span className="image-title" style={{ color: "var(--primary)" }}>
                  3. Grad-CAM Heatmap
                </span>
                <span className="badge" style={{ background: "var(--surface-2)" }}>JET Colormap</span>
              </div>
              <div
                className="image-frame"
                onClick={() =>
                  setActiveLightbox({
                    src: record.heatmap_image || "",
                    title: "Grad-CAM Heatmap",
                    desc: "Peta bobot gradien murni colormap JET (Merah: aktivasi tinggi)",
                  })
                }
                title="Klik untuk memperbesar"
              >
                {record.heatmap_image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={record.heatmap_image} alt="Grad-CAM Heatmap" />
                ) : (
                  <div className="no-image-placeholder">Heatmap tidak tersedia</div>
                )}
                <div className="expand-overlay no-print">
                  <span> Klik untuk Perbesar</span>
                </div>
              </div>
              <div className="image-caption">
                Peta perhatian gradien aktivasi murni; spektrum merah menandai region paling memengaruhi keputusan AI.
              </div>
            </div>
          )}
        </div>
      </section>

      {/* =================================================================== */}
      {/* SECTION D: ACTIVATION ANALYSIS                                      */}
      {/* =================================================================== */}
      <section className="report-section">
        <div className="section-title-wrap">
          <span className="eyebrow">Analisis Distribusi Region</span>
          <h2 className="section-heading">Distribusi Aktivasi 9 Area Citra</h2>
        </div>

        <div className="activation-analysis-layout">
          {/* Kolom Kiri: Ringkasan Pola Aktivasi */}
          <div className="pattern-summary-card">
            <h3 className="pattern-summary-heading">Ringkasan Pola Aktivasi Visual</h3>

            <div className="pattern-item-block">
              <span className="pattern-item-label">Area Dominan:</span>
              <div className="pattern-item-highlight">
                {dominantRegion ? (
                  <>
                    <b style={{ textTransform: "capitalize", color: "var(--primary)" }}>{dominantRegion}</b>
                    <span className="meta-pill" style={{ marginLeft: 8 }}>
                      {report?.visualization?.dominant_activation_percentage?.toFixed(2)}%
                    </span>
                  </>
                ) : (
                  <span className="muted">Belum terhitung</span>
                )}
              </div>
            </div>

            <div className="pattern-item-block">
              <span className="pattern-item-label">Pola Distribusi:</span>
              <div className="pattern-item-highlight">
                <span className="badge" style={{ fontSize: 13, background: "var(--surface-2)", color: "var(--primary)" }}>
                  {report?.visualization?.activation_pattern || "Standard"}
                </span>
              </div>
              <p className="pattern-item-desc">
                {report?.visualization?.activation_pattern_description ||
                  "Distribusi aktivasi terpusat pada beberapa area utama visual citra."}
              </p>
            </div>

            <div className="pattern-item-block">
              <span className="pattern-item-label">Cakupan Area Kuat (&ge; 50% max):</span>
              <div className="pattern-item-highlight mono" style={{ fontSize: 18, fontWeight: 700 }}>
                {report?.visualization?.coverage_percentage !== undefined && report?.visualization?.coverage_percentage !== null
                  ? `${report.visualization.coverage_percentage.toFixed(2)}%`
                  : "-"}
              </div>
            </div>

            <div className="pattern-disclaimer-box">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span>
                Persentase aktivasi Grad-CAM mencerminkan bobot perhatian visual komputasi model, bukan ukuran lesi patologis atau luas area penyebaran penyakit.
              </span>
            </div>
          </div>

          {/* Kolom Kanan: Horizontal Progress Bars untuk 9 Region */}
          <div className="region-bars-card">
            <h3 className="region-bars-heading">Distribusi Relatif Sembilan Region Citra</h3>
            <p className="region-bars-subtitle">
              Diurutkan berdasarkan besaran persentase aktivasi gradien pada kisi 3&times;3 citra endoskopi:
            </p>

            <div className="region-bars-list">
              {regionList.map((item, idx) => {
                const isDominant = dominantRegion && item.region.toLowerCase() === dominantRegion.toLowerCase();
                return (
                  <div key={item.region || idx} className={`region-bar-item ${isDominant ? "dominant" : ""}`}>
                    <div className="region-bar-labels">
                      <span className="region-bar-name">
                        {/* {isDominant && <span className="dominant-star" title="Area Dominan">★ </span>} */}
                        {item.region}
                        {isDominant && <span className="dominant-badge">Dominan</span>}
                      </span>
                      <span className="region-bar-value mono">
                        {item.percentage.toFixed(2)}%
                      </span>
                    </div>

                    <div className="region-progress-track">
                      <div
                        className={`region-progress-fill ${isDominant ? "dominant" : ""}`}
                        style={{ width: `${Math.min(100, item.percentage * 2)}%` }} // Scaled relative visually
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* =================================================================== */}
      {/* SECTION F: INTERPRETATION                                           */}
      {/* =================================================================== */}
      <section className="report-section">
        <div className="section-title-wrap">
          <span className="eyebrow">Interpretasi Medis Komprehensif</span>
          <h2 className="section-heading">Analisis Terintegrasi & Catatan Klinis</h2>
        </div>

        <div className="interpretation-grid">
          {/* 1. Ringkasan Sintesis */}
          <div className="interpretation-card full-width">
            <div className="interp-header">              
              <h3 className="interp-title">Ringkasan Analisis Medis (Summary)</h3>
            </div>
            <p className="interp-content-lead">
              {report?.explanation?.summary ||
                `Model ${record.model_name || "EfficientNetB5"} mengklasifikasikan citra sebagai ${
                  isGerd ? "GERD" : "GERD_Normal"
                } dengan tingkat keyakinan ${pct(record.confidence)}.`}
            </p>
          </div>

          {/* 2. Interpretasi Prediksi & Probabilitas */}
          <div className="interpretation-card">
            <div className="interp-header">
              <h3 className="interp-title">Interpretasi Prediksi & Probabilitas Model</h3>
            </div>
            <div className="interp-body-text">
              <p>
                <b>Klasifikasi:</b>{" "}
                {report?.explanation?.prediction_interpretation ||
                  `EfficientNetB5 mengklasifikasikan citra ke dalam kategori ${isGerd ? "GERD" : "GERD_Normal"}.`}
              </p>
              <p style={{ marginTop: 8 }}>
                <b>Probabilitas:</b>{" "}
                {report?.explanation?.probability_interpretation ||
                  `Probabilitas GERD sebesar ${pct(record.probabilities?.GERD)}, sedangkan GERD_Normal sebesar ${pct(
                    record.probabilities?.GERD_Normal ?? record.probabilities?.Normal
                  )}.`}
              </p>
            </div>
          </div>

          {/* 3. Interpretasi Visual Grad-CAM */}
          <div className="interpretation-card">
            <div className="interp-header">
              <h3 className="interp-title">Interpretasi Visual Grad-CAM</h3>
            </div>
            <div className="interp-body-text">
              <p>
                {report?.explanation?.visual_interpretation ||
                  "Aktivasi Grad-CAM menunjukkan fitur visual kunci yang paling diperhatikan oleh model saat memproses citra endoskopi."}
              </p>
            </div>
          </div>

          {/* 4. Catatan Klinis Dokter */}
          <div className="interpretation-card full-width clinical-note-card">
            <div className="interp-header">              
              <h3 className="interp-title">Catatan Klinis Dokter & Batasan Sistem</h3>
            </div>
            <p className="interp-content-body">
              {report?.explanation?.clinical_note ||
                "Hasil ini merupakan keluaran model klasifikasi berbasis citra. Grad-CAM menunjukkan area visual yang berkontribusi terhadap prediksi model dan tidak dapat dianggap sebagai bukti langsung adanya lesi, inflamasi, atau kelainan anatomi. Penentuan temuan klinis dan keputusan terapi medis memerlukan evaluasi langsung oleh dokter spesialis."}
            </p>
          </div>
        </div>
      </section>

      {/* =================================================================== */}
      {/* SECTION G: DISCLAIMER                                               */}
      {/* =================================================================== */}
      <footer className="report-disclaimer-footer">
        {/* <div className="disclaimer-icon">⚠️</div> */}
        <div className="disclaimer-content">
          <b>Batas Tanggung Jawab Klinis (Medical Disclaimer):</b>
          <p>
            Hasil keluaran kecerdasan buatan ini merupakan alat bantu penyaringan (*clinical screening tool*) berbasis citra dan bukan diagnosis medis mandiri definitif. Grad-CAM merefleksikan region aktivasi fitur matematis model dan tidak membuktikan lokasi lesi patologis atau kelainan anatomi secara definitif. Keputusan klinis, penetapan diagnosis akhir, dan rencana penatalaksanaan terapi pasien tetap menjadi tanggung jawab penuh dokter spesialis gastroenterologi / endoskopis.
          </p>
        </div>
      </footer>

      {/* =================================================================== */}
      {/* LIGHTBOX MODAL (EXPAND IMAGE)                                       */}
      {/* =================================================================== */}
      {activeLightbox && (
        <div
          className="lightbox-backdrop no-print"
          onClick={() => setActiveLightbox(null)}
          role="dialog"
          aria-modal="true"
        >
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <div className="lightbox-header">
              <div>
                <h4 className="lightbox-title">{activeLightbox.title}</h4>
                <p className="lightbox-desc">{activeLightbox.desc}</p>
              </div>
              <button
                type="button"
                className="btn btn-ghost lightbox-close-btn"
                onClick={() => setActiveLightbox(null)}
                aria-label="Tutup pratinjau"
              >
                ✕ Tutup
              </button>
            </div>

            <div className="lightbox-image-box">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={activeLightbox.src} alt={activeLightbox.title} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
