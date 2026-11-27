/* eslint-disable @next/next/no-img-element */
"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import ResultDialog from "@/components/ResultDialog";
import {
  clearHistory,
  DEFAULT_PROFILE,
  deleteRecord,
  deleteRecords,
  HISTORY_KEY,
  PROFILE_KEY,
  useStored,
  type HistoryRecord,
} from "@/lib/store";

const PER_PAGE = 5;
type Filter = "all" | "GERD" | "Normal";

export default function Riwayat() {
  const [rows, setRows, ready] = useStored<HistoryRecord[]>(HISTORY_KEY, []);
  const [profile] = useStored(PROFILE_KEY, DEFAULT_PROFILE);
  const [filter, setFilter] = useState<Filter>("all");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(1);
  const [open, setOpen] = useState<HistoryRecord | null>(null);

  // State untuk voting / checklist riwayat yang ingin dihapus
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const gerd = rows.filter((r) => r.label === "GERD").length;
  const shown = useMemo(
    () =>
      rows.filter(
        (r) =>
          (filter === "all" || r.label === filter) &&
          r.patientId.toLowerCase().includes(q.trim().toLowerCase()),
      ),
    [rows, filter, q],
  );
  const pages = Math.max(1, Math.ceil(shown.length / PER_PAGE));
  const current = Math.min(page, pages);
  const slice = shown.slice((current - 1) * PER_PAGE, current * PER_PAGE);
  const share = (n: number) =>
    rows.length ? `${((n / rows.length) * 100).toFixed(1)}%` : "0%";

  // Cek apakah seluruh baris di halaman aktif sedang terpilih
  const isAllCurrentSelected =
    slice.length > 0 && slice.every((r) => selectedIds.includes(r.id));

  function toggleSelectAllCurrent() {
    if (isAllCurrentSelected) {
      const sliceIds = new Set(slice.map((r) => r.id));
      setSelectedIds((prev) => prev.filter((id) => !sliceIds.has(id)));
    } else {
      const merged = new Set([...selectedIds, ...slice.map((r) => r.id)]);
      setSelectedIds(Array.from(merged));
    }
  }

  function toggleSelect(id: string) {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id],
    );
  }

  // 1. Hapus salah satu riwayat
  function handleDeleteOne(rec: HistoryRecord) {
    const confirmMsg = `Hapus riwayat pemeriksaan untuk pasien "${rec.patientId}" (Berkas: ${rec.file})?`;
    if (window.confirm(confirmMsg)) {
      deleteRecord(rec.id);
      setRows((prev) => prev.filter((r) => r.id !== rec.id));
      setSelectedIds((prev) => prev.filter((id) => id !== rec.id));
      if (open?.id === rec.id) setOpen(null);
    }
  }

  // 2. Hapus riwayat yang dipilih (voted/checked)
  function handleDeleteSelected() {
    if (!selectedIds.length) return;
    const confirmMsg = `Apakah Anda yakin ingin menghapus ${selectedIds.length} riwayat yang dipilih? Tindakan ini tidak dapat dibatalkan.`;
    if (window.confirm(confirmMsg)) {
      deleteRecords(selectedIds);
      const idSet = new Set(selectedIds);
      setRows((prev) => prev.filter((r) => !idSet.has(r.id)));
      setSelectedIds([]);
      if (open && idSet.has(open.id)) setOpen(null);
    }
  }

  // 3. Hapus seluruh riwayat
  function handleDeleteAll() {
    if (!rows.length) return;
    const confirmMsg = `Apakah Anda yakin ingin menghapus SELURUH riwayat (${rows.length} data pemeriksaan)? Tindakan ini tidak dapat dibatalkan.`;
    if (window.confirm(confirmMsg)) {
      clearHistory();
      setRows([]);
      setSelectedIds([]);
      setOpen(null);
    }
  }

  function exportCsv() {
    const esc = (v: string | number) => `"${String(v).replace(/"/g, '""')}"`;
    const lines = [
      ["id_pasien", "waktu", "usia", "jk", "berkas", "label", "keyakinan"].join(
        ",",
      ),
    ].concat(
      shown.map((r) =>
        [
          r.patientId,
          r.at,
          r.age,
          r.sex,
          r.file,
          r.label,
          r.confidence.toFixed(4),
        ]
          .map(esc)
          .join(","),
      ),
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(
      new Blob([lines.join("\n")], { type: "text/csv" }),
    );
    a.download = "riwayat-analisis.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const chip = (f: Filter, label: string, n: number) => (
    <button
      key={f}
      className="chip"
      aria-pressed={filter === f}
      onClick={() => {
        setFilter(f);
        setPage(1);
      }}
    >
      {label} ({n})
    </button>
  );

  return (
    <div className="wrap" style={{ padding: "48px var(--pad) 80px" }}>
      {/* Header Halaman */}
      <div
        className="row"
        style={{ justifyContent: "space-between", alignItems: "flex-end" }}
      >
        <div>
          <p className="eyebrow">Arsip pemeriksaan</p>
          <h1 style={{ fontSize: "clamp(28px,4vw,40px)", marginTop: 12 }}>
            Riwayat analisis
          </h1>
          <p className="muted" style={{ marginTop: 12 }}>
            Hasil yang Anda simpan dari modul analisis, tersimpan di browser
            ini.
          </p>
        </div>
        <div className="row no-print" style={{ gap: 8 }}>
          <button
            className="btn btn-ghost"
            onClick={exportCsv}
            disabled={!shown.length}
          >
            Ekspor CSV
          </button>
          <button className="btn btn-ghost" onClick={() => window.print()}>
            Cetak
          </button>
          {rows.length > 0 && (
            <button
              className="btn btn-ghost"
              style={{ color: "var(--danger)", borderColor: "var(--danger)" }}
              onClick={handleDeleteAll}
              title="Hapus seluruh riwayat yang tersimpan"
            >
              Hapus Semua
            </button>
          )}
          <Link href="/analisis" className="btn btn-primary">
            + Analisis baru
          </Link>
        </div>
      </div>

      {/* Ringkasan Statistik */}
      <div className="stats" style={{ marginTop: 32 }}>
        <div className="panel">
          <span className="label muted">Total analisis</span>
          <b>{rows.length}</b>
        </div>
        <div className="panel">
          <span className="label muted">Prediksi GERD</span>
          <b style={{ color: "var(--warn)" }}>{gerd}</b>
          <span className="small muted">{share(gerd)} dari total</span>
        </div>
        <div className="panel">
          <span className="label muted">Prediksi Normal</span>
          <b style={{ color: "var(--ok)" }}>{rows.length - gerd}</b>
          <span className="small muted">
            {share(rows.length - gerd)} dari total
          </span>
        </div>
      </div>

      {/* Filter & Pencarian */}
      <div
        className="row no-print"
        style={{ margin: "32px 0 16px", justifyContent: "space-between" }}
      >
        <div className="chips">
          {chip("all", "Semua", rows.length)}
          {chip("GERD", "GERD", gerd)}
          {chip("Normal", "Normal", rows.length - gerd)}
        </div>
        <input
          className="input"
          style={{ maxWidth: 260 }}
          placeholder="Cari ID pasien"
          aria-label="Cari ID pasien"
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setPage(1);
          }}
        />
      </div>

      {/* Bar Tindakan Seleksi (Muncul saat ada riwayat yang dipilih / voted) */}
      {selectedIds.length > 0 && (
        <div
          className="notice no-print"
          style={{
            marginBottom: 16,
            background: "var(--warn-bg)",
            borderColor: "var(--warn)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 12,
            padding: "10px 16px",
          }}
        >
          <div className="row" style={{ gap: 8 }}>
            <span style={{ fontSize: 16 }}>🗳️</span>
            <b>{selectedIds.length} riwayat dipilih untuk dihapus</b>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <button
              type="button"
              className="btn btn-ghost"
              style={{ padding: "6px 12px", minHeight: 32, fontSize: 12 }}
              onClick={() => setSelectedIds([])}
            >
              Batal Pilih
            </button>
            <button
              type="button"
              className="btn btn-primary"
              style={{
                padding: "6px 14px",
                minHeight: 32,
                fontSize: 12,
                background: "var(--danger)",
                borderColor: "var(--danger)",
              }}
              onClick={handleDeleteSelected}
            >
              🗑️ Hapus Terpilih ({selectedIds.length})
            </button>
          </div>
        </div>
      )}

      {/* Tabel Riwayat */}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {/* Checkbox Pilih Semua di Halaman Ini */}
              <th
                style={{ width: 44, textAlign: "center" }}
                className="no-print"
              >
                <input
                  type="checkbox"
                  aria-label="Pilih semua baris di halaman ini"
                  checked={isAllCurrentSelected}
                  onChange={toggleSelectAllCurrent}
                  disabled={!slice.length}
                  style={{
                    accentColor: "var(--primary)",
                    width: 16,
                    height: 16,
                    cursor: slice.length ? "pointer" : "default",
                  }}
                />
              </th>
              <th>ID pasien</th>
              <th>Waktu</th>
              <th>Citra</th>
              <th>Hasil</th>
              <th>Keyakinan</th>
              <th className="no-print" style={{ textAlign: "right" }}>
                Aksi
              </th>
            </tr>
          </thead>
          <tbody>
            {slice.map((r) => {
              const isChecked = selectedIds.includes(r.id);
              return (
                <tr
                  key={r.id}
                  style={{
                    background: isChecked ? "var(--surface-2)" : undefined,
                  }}
                >
                  {/* Checkbox Voting Baris */}
                  <td style={{ textAlign: "center" }} className="no-print">
                    <input
                      type="checkbox"
                      aria-label={`Pilih riwayat pasien ${r.patientId}`}
                      checked={isChecked}
                      onChange={() => toggleSelect(r.id)}
                      style={{
                        accentColor: "var(--primary)",
                        width: 16,
                        height: 16,
                        cursor: "pointer",
                      }}
                    />
                  </td>
                  <td>
                    <b>{r.patientId}</b>
                    <div className="small muted">
                      {r.sex} · {r.age} th
                    </div>
                  </td>
                  <td>
                    {new Date(r.at).toLocaleDateString("id-ID", {
                      day: "2-digit",
                      month: "short",
                      year: "numeric",
                    })}
                    <div className="small muted">
                      {new Date(r.at).toLocaleTimeString("id-ID", {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </div>
                  </td>
                  <td>
                    {r.thumb ? (
                      <img className="thumb" src={r.thumb} alt="" />
                    ) : (
                      <span className="muted">-</span>
                    )}
                    <div className="small muted mono">{r.file}</div>
                  </td>
                  <td>
                    <span
                      className={`badge ${r.label === "GERD" ? "gerd" : "normal"}`}
                    >
                      {r.label}
                    </span>
                  </td>
                  <td className="mono">{(r.confidence * 100).toFixed(1)}%</td>
                  {/* Tombol Aksi per Baris */}
                  <td className="no-print" style={{ textAlign: "right" }}>
                    <div
                      className="row"
                      style={{ justifyContent: "flex-end", gap: 6 }}
                    >
                      <button
                        type="button"
                        className="btn btn-ghost"
                        style={{
                          padding: "4px 10px",
                          minHeight: 30,
                          fontSize: 12,
                        }}
                        onClick={() => setOpen(r)}
                      >
                        Detail
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        style={{
                          padding: "4px 8px",
                          minHeight: 30,
                          fontSize: 12,
                          color: "var(--danger)",
                        }}
                        onClick={() => handleDeleteOne(r)}
                        title="Hapus riwayat ini"
                      >
                        Hapus
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {ready && !slice.length && (
              <tr>
                <td
                  colSpan={7}
                  className="muted"
                  style={{ padding: 40, textAlign: "center" }}
                >
                  {rows.length ? (
                    "Tidak ada hasil yang cocok dengan pencarian/filter."
                  ) : (
                    <>
                      Belum ada riwayat.{" "}
                      <Link href="/analisis">Mulai analisis pertama</Link>.
                    </>
                  )}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination & Status Footer */}
      <div
        className="row no-print"
        style={{ justifyContent: "space-between", marginTop: 16 }}
      >
        <span className="small muted">
          Menampilkan {slice.length} dari {shown.length} rekam analisis
          {selectedIds.length > 0 && ` (${selectedIds.length} dipilih)`}
        </span>
        <div className="row">
          <button
            type="button"
            className="btn btn-ghost"
            disabled={current <= 1}
            onClick={() => setPage(current - 1)}
          >
            Sebelumnya
          </button>
          <span className="small muted">
            {current} / {pages}
          </span>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={current >= pages}
            onClick={() => setPage(current + 1)}
          >
            Berikutnya
          </button>
        </div>
      </div>

      {open && (
        <ResultDialog
          record={open}
          threshold={profile.threshold}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}
