// test-frontend-integration.mjs
// Pengujian Integrasi Frontend GERDGuard dengan Backend FastAPI & Grad-CAM (Tahap 5)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BACKEND_URL = "http://127.0.0.1:8000";
const TEST_DIR = path.resolve(__dirname, "../backend/testing_guard");

const GERD_IMG_PATH = path.join(TEST_DIR, "valid_endoscopy", "lbaidcw.jpg");
const NORMAL_IMG_PATH = path.join(TEST_DIR, "valid_endoscopy", "Gerd Normal_3579.jpg");
const FOREIGN_IMG_PATH = path.join(TEST_DIR, "foreign_images", "images.jpg");

const results = [];

function test(name, fn) {
  return async () => {
    console.log(`\n=================================================================`);
    console.log(`TEST: ${name}`);
    console.log(`=================================================================`);
    try {
      await fn();
      results.push({ name, pass: true, err: "" });
      console.log(`  ✓ PASS`);
    } catch (e) {
      results.push({ name, pass: false, err: e.message });
      console.log(`  ✗ FAIL: ${e.message}`);
      console.log(e.stack);
    }
  };
}

// Helper untuk fetch backend sesuai logika lib/api.ts
async function callPredict(filePath, mimeType = "image/jpeg", backendBase = BACKEND_URL) {
  const fileBytes = fs.readFileSync(filePath);
  const blob = new Blob([fileBytes], { type: mimeType });
  const form = new FormData();
  form.append("file", blob, path.basename(filePath));

  const res = await fetch(`${backendBase}/predict`, {
    method: "POST",
    body: form,
  });

  const body = await res.json().catch(() => null);
  return { status: res.status, body };
}

// Helper simulasi normalize() dari lib/api.ts
function simulateNormalize(body) {
  if (!body) throw new Error("Body respons kosong");
  const statusStr = String(body.status || body.prediction || "").toUpperCase();
  const gateStatus = String(body.gate_status || "").toUpperCase();

  if (statusStr === "DITOLAK" || gateStatus === "DITOLAK") {
    const err = new Error(body.reason || "Citra ditolak oleh Image Type Gate.");
    err.status = 422;
    throw err;
  }

  const labelRaw = body.predicted_class || body.label || body.prediction || "GERD_Normal";
  const label = /normal/i.test(labelRaw) ? "GERD_Normal" : "GERD";

  return {
    status: label,
    predicted_class: label,
    label,
    probability_gerd: body.probability_gerd ?? 0,
    probability_normal: body.probability_normal ?? 0,
    confidence: body.confidence ?? 0,
    gate_status: body.gate_status || "LOLOS_GATE",
    gate_score: body.gate_score,
    model_name: body.model_name || "EfficientNetB5",
    original_image: body.original_image || null,
    heatmap_image: body.heatmap_image || null,
    overlay_image: body.overlay_image || null,
    report: body.report || null,
  };
}

async function runAll() {
  console.log("Memulai pengujian integrasi Frontend GERDGuard...");

  // 1. Uji komunikasi dasar & endpoint /health
  await test("1. Uji komunikasi frontend dan backend via /health", async () => {
    const res = await fetch(`${BACKEND_URL}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.status !== "ok") throw new Error(`Status bukan ok: ${data.status}`);
    if (!data.model_loaded) throw new Error("Model belum dimuat");
    if (!data.image_type_gate_loaded) throw new Error("Gate belum dimuat");
    console.log(`  Backend connected: ${data.model_name} (Gate threshold: ${data.gate_threshold})`);
  })();

  // 2. Uji upload gambar dengan FormData
  await test("2. Uji upload gambar menggunakan FormData multipart/form-data", async () => {
    if (!fs.existsSync(GERD_IMG_PATH)) throw new Error(`File tidak ditemukan: ${GERD_IMG_PATH}`);
    const { status, body } = await callPredict(GERD_IMG_PATH);
    if (status !== 200) throw new Error(`Status code ${status}: ${JSON.stringify(body)}`);
    if (!body.model_name) throw new Error("Respons tidak memiliki field model_name");
    console.log(`  Upload FormData berhasil, response diterima dengan status 200.`);
  })();

  // 3. Uji gambar yang diterima Gate
  await test("3. Uji gambar endoskopi yang diterima Gate (LOLOS_GATE)", async () => {
    const { status, body } = await callPredict(GERD_IMG_PATH);
    if (status !== 200) throw new Error(`Status ${status}`);
    if (body.gate_status !== "LOLOS_GATE") throw new Error(`gate_status: ${body.gate_status}`);
    if (body.gate_score >= 0.01) throw new Error(`gate_score terlalu tinggi: ${body.gate_score}`);
    console.log(`  Gate status: ${body.gate_status}, score non-endoskopi: ${body.gate_score}`);
  })();

  // 4. Uji gambar yang ditolak Gate
  await test("4. Uji gambar non-endoskopi yang ditolak Gate (DITOLAK)", async () => {
    const { status, body } = await callPredict(FOREIGN_IMG_PATH);
    if (status !== 200) throw new Error(`Status ${status}`);
    if (body.status !== "DITOLAK" || body.gate_status !== "DITOLAK") {
      throw new Error(`Expected DITOLAK, got status=${body.status}, gate_status=${body.gate_status}`);
    }
    if (body.predicted_class !== null || body.heatmap_image !== null || body.report !== null) {
      throw new Error("EfficientNetB5, Grad-CAM, atau Report tidak boleh dijalankan untuk gambar yang ditolak Gate");
    }
    // Uji apakah normalize() melempar ApiError 422 untuk UI alert
    let rejectedCaught = false;
    try {
      simulateNormalize(body);
    } catch (e) {
      if (e.status === 422) rejectedCaught = true;
    }
    if (!rejectedCaught) throw new Error("simulateNormalize tidak melempar error status 422 untuk penolakan Gate");
    console.log(`  Ditolak oleh gate: "${body.reason}", B5/Grad-CAM/Report dinonaktifkan: OK`);
  })();

  // 5. Uji tampilan hasil GERD
  await test("5. Uji tampilan hasil klasifikasi GERD (lbaidcw.jpg)", async () => {
    const { body } = await callPredict(GERD_IMG_PATH);
    const norm = simulateNormalize(body);
    if (norm.predicted_class !== "GERD") throw new Error(`Expected GERD, got ${norm.predicted_class}`);
    if (norm.probability_gerd < 0.5) throw new Error(`P(GERD) rendah: ${norm.probability_gerd}`);
    if (norm.confidence < 0.5) throw new Error(`Confidence rendah: ${norm.confidence}`);
    console.log(`  Predicted: ${norm.predicted_class}, P(GERD): ${(norm.probability_gerd * 100).toFixed(2)}%, Confidence: ${(norm.confidence * 100).toFixed(2)}%`);
  })();

  // 6. Uji tampilan hasil GERD_Normal
  await test("6. Uji tampilan hasil klasifikasi GERD_Normal (Gerd Normal_3579.jpg)", async () => {
    const { body } = await callPredict(NORMAL_IMG_PATH);
    const norm = simulateNormalize(body);
    if (norm.predicted_class !== "GERD_Normal") throw new Error(`Expected GERD_Normal, got ${norm.predicted_class}`);
    if (norm.probability_normal < 0.5) throw new Error(`P(Normal) rendah: ${norm.probability_normal}`);
    console.log(`  Predicted: ${norm.predicted_class}, P(GERD_Normal): ${(norm.probability_normal * 100).toFixed(2)}%, Confidence: ${(norm.confidence * 100).toFixed(2)}%`);
  })();

  // 7. Uji tampilan heatmap dan overlay Grad-CAM
  await test("7. Uji validitas gambar Base64 / data URI (Original, Heatmap, Overlay)", async () => {
    const { body } = await callPredict(GERD_IMG_PATH);
    const norm = simulateNormalize(body);

    if (!norm.original_image || !norm.original_image.startsWith("data:image/")) {
      throw new Error("original_image tidak valid atau kosong");
    }
    if (!norm.heatmap_image || !norm.heatmap_image.startsWith("data:image/jpeg;base64,")) {
      throw new Error("heatmap_image tidak valid atau kosong");
    }
    if (!norm.overlay_image || !norm.overlay_image.startsWith("data:image/jpeg;base64,")) {
      throw new Error("overlay_image tidak valid atau kosong");
    }

    // Periksa panjang payload base64
    const heatmapPayload = norm.heatmap_image.split(",")[1];
    const overlayPayload = norm.overlay_image.split(",")[1];
    const heatmapBuf = Buffer.from(heatmapPayload, "base64");
    const overlayBuf = Buffer.from(overlayPayload, "base64");

    if (heatmapBuf.length < 1000) throw new Error(`Heatmap payload terlalu kecil (${heatmapBuf.length} bytes)`);
    if (overlayBuf.length < 1000) throw new Error(`Overlay payload terlalu kecil (${overlayBuf.length} bytes)`);

    console.log(`  original_image: ${norm.original_image.slice(0, 30)}... (${norm.original_image.length} chars)`);
    console.log(`  heatmap_image : ${norm.heatmap_image.slice(0, 30)}... (${heatmapBuf.length} bytes)`);
    console.log(`  overlay_image : ${norm.overlay_image.slice(0, 30)}... (${overlayBuf.length} bytes)`);
  })();

  // 8. Uji error ketika backend mati
  await test("8. Uji penanganan error ketika backend tidak tersedia (port offline)", async () => {
    const DEAD_PORT_URL = "http://127.0.0.1:59999";
    let caught = false;
    let errorMsg = "";

    try {
      await callPredict(GERD_IMG_PATH, "image/jpeg", DEAD_PORT_URL);
    } catch (e) {
      caught = true;
      errorMsg = "Backend tidak tersedia. Pastikan server FastAPI aktif di port 8000.";
    }

    if (!caught) throw new Error("Permintaan ke backend mati tidak melempar exception");
    console.log(`  Koneksi ke ${DEAD_PORT_URL} gagal ditangkap dengan pesan: "${errorMsg}"`);
  })();

  // 9. Uji struktur Patient Report Generator (v2.0)
  await test("9. Uji struktur Patient Report Generator (v2.0)", async () => {
    const { body } = await callPredict(GERD_IMG_PATH);
    if (!body.report) throw new Error("Field report tidak ada dalam response");
    if (body.report.report_version !== "2.0") throw new Error(`Versi report bukan 2.0: ${body.report.report_version}`);
    if (!body.report.explanation?.summary) throw new Error("Report explanation summary kosong");
    if (!body.report.classification?.confidence_level) throw new Error("Confidence level kosong");
    if (!body.report.visualization?.region_distribution) throw new Error("Region distribution kosong");
    console.log(`  Report v${body.report.report_version} validated: ${body.report.classification.confidence_level} confidence`);
    console.log(`  Summary: ${body.report.explanation.summary.slice(0, 80)}...`);
  })();

  // Ringkasan
  console.log(`\n=================================================================`);
  console.log("RINGKASAN PENGUJIAN INTEGRASI FRONTEND (TAHAP 5)");
  console.log(`=================================================================`);
  const passed = results.filter((r) => r.pass).length;
  const failed = results.filter((r) => !r.pass).length;
  for (const r of results) {
    const status = r.pass ? "✓ PASS" : `✗ FAIL: ${r.err}`;
    console.log(`  ${status.padEnd(12)}  ${r.name}`);
  }
  console.log(`\nTotal: ${passed} passed, ${failed} failed, ${results.length} total`);

  if (failed > 0) {
    process.exit(1);
  } else {
    console.log("\n🎉 Seluruh 9 skenario pengujian integrasi frontend BERHASIL!");
    process.exit(0);
  }
}

runAll();
