// lib/api.ts
// Format respons & komunikasi API backend FastAPI GERDGuard (EfficientNetB5 + Image Type Gate B4 + Grad-CAM)

export type Label = "GERD" | "GERD_Normal" | "Normal";

export type RegionDistributionItem = {
  region: string;
  score: number;
  percentage: number;
};

export type HeatmapVisualizationAnalysis = {
  available: boolean;
  dominant_region: string | null;
  dominant_activation_percentage?: number | null;
  activation_pattern: "localized" | "multifocal" | "distributed" | "zero_activation" | "unavailable" | string;
  activation_pattern_description?: string;
  coverage_percentage: number | null;
  region_distribution: RegionDistributionItem[];
  top_regions: RegionDistributionItem[];
  explanation: string;
};

export type ClassificationAnalysis = {
  predicted_class: "GERD" | "GERD_Normal";
  p_gerd: number;
  p_normal: number;
  confidence: number;
  confidence_percentage: number;
  confidence_level: "Tinggi" | "Sedang" | "Rendah";
  confidence_description: string;
  explanation: string;
  detail: string;
};

export type ImageGateAnalysis = {
  status: string;
  explanation: string;
};

export type ClinicalFindings = {
  lesion_detection: string;
  erosion_detection: string;
  inflammation_detection: string;
  ulcer_detection: string;
  la_grade: string;
  explanation: string;
};

export type ReportExplanation = {
  summary: string;
  prediction_interpretation: string;
  probability_interpretation: string;
  visual_interpretation: string;
  clinical_note: string;
};

export type PatientReport = {
  report_version: string;
  classification: ClassificationAnalysis;
  image_gate: ImageGateAnalysis;
  visualization: HeatmapVisualizationAnalysis;
  findings: ClinicalFindings;
  explanation: ReportExplanation;
};

export type PredictionResult = {
  status: string; // "GERD" | "GERD_Normal" | "DITOLAK"
  predicted_class: "GERD" | "GERD_Normal" | null;
  label: Label;
  probability_gerd: number | null;
  probability_normal: number | null;
  confidence: number; // 0..1
  gate_status: "LOLOS_GATE" | "DITOLAK" | string;
  gate_score: number | null;
  model_name: string;
  original_image: string | null;
  heatmap_image: string | null;
  overlay_image: string | null;
  reason?: string | null;
  probabilities?: Record<string, number>; // 0..1
  report?: PatientReport | null;
};

// Alias untuk kompatibilitas ke belakang
export type Prediction = PredictionResult;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

const PREDICT_PATH = process.env.NEXT_PUBLIC_PREDICT_PATH || "/predict";
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const toUnit = (n: number) => (n > 1 ? n / 100 : n);

/**
 * Normalisasi payload respons dari backend FastAPI.
 */
export function normalize(raw: unknown): PredictionResult {
  const r = (raw as Record<string, any>) ?? {};

  // Cek apakah ditolak oleh gate
  const statusStr = String(r.status || r.prediction || "").toUpperCase();
  const gateStatus = String(r.gate_status || "").toUpperCase();

  if (statusStr === "DITOLAK" || gateStatus === "DITOLAK") {
    throw new ApiError(
      422,
      r.reason || r.alasan || "Citra ditolak oleh Image Type Gate: terdeteksi bukan foto endoskopi medis."
    );
  }

  const labelRaw = r.predicted_class || r.label || r.prediction || "GERD_Normal";
  const label: Label = /normal/i.test(labelRaw) ? "GERD_Normal" : "GERD";

  // Probabilitas
  let probGerd: number | null = null;
  let probNormal: number | null = null;

  if (typeof r.probability_gerd === "number") {
    probGerd = toUnit(r.probability_gerd);
  }
  if (typeof r.probability_normal === "number") {
    probNormal = toUnit(r.probability_normal);
  }

  // Fallback jika dikirim dalam objek probabilities
  if (probGerd === null && r.probabilities?.GERD !== undefined) {
    probGerd = toUnit(Number(r.probabilities.GERD));
  }
  if (probNormal === null && (r.probabilities?.GERD_Normal !== undefined || r.probabilities?.Normal !== undefined)) {
    probNormal = toUnit(Number(r.probabilities.GERD_Normal ?? r.probabilities.Normal));
  }

  // Confidence
  let conf = typeof r.confidence === "number" ? toUnit(r.confidence) : 0;
  if (!conf) {
    conf = label === "GERD" ? (probGerd ?? 0.5) : (probNormal ?? 0.5);
  }

  const probabilities: Record<string, number> = {
    GERD: probGerd ?? (label === "GERD" ? conf : 1 - conf),
    GERD_Normal: probNormal ?? (label === "GERD_Normal" ? conf : 1 - conf),
  };

  return {
    status: label,
    predicted_class: label === "GERD" ? "GERD" : "GERD_Normal",
    label,
    probability_gerd: probabilities.GERD,
    probability_normal: probabilities.GERD_Normal,
    confidence: conf,
    gate_status: (r.gate_status as string) || "LOLOS_GATE",
    gate_score: typeof r.gate_score === "number" ? r.gate_score : null,
    model_name: r.model_name || "EfficientNetB5",
    original_image: r.original_image || null,
    heatmap_image: r.heatmap_image || null,
    overlay_image: r.overlay_image || null,
    reason: r.reason || null,
    probabilities,
    report: (r.report as PatientReport) || null,
  };
}

function errorMessage(body: any, status: number): string {
  const d = body?.detail ?? body?.message ?? body?.error ?? body?.reason;
  if (typeof d === "string") return d;
  if (d && typeof d === "object") return d.message ?? d.pesan ?? d.reason ?? JSON.stringify(d);
  if (status === 415) return "Format gambar tidak didukung. Gunakan JPG, PNG, atau BMP.";
  if (status === 413) return "Ukuran gambar melebihi batas maksimal 10 MB.";
  if (status === 400) return "File gambar kosong atau rusak.";
  if (status === 503) return "Model atau layanan analisis belum siap di backend.";
  if (status === 500) return "Terjadi kesalahan internal pada server analisis atau Grad-CAM.";
  return `Server menjawab dengan status ${status}.`;
}

/**
 * Mengirim gambar ke endpoint POST /predict backend FastAPI menggunakan FormData.
 */
export async function predict(file: File): Promise<PredictionResult> {
  const form = new FormData();
  form.append("file", file);

  let res: Response | null = null;
  let fetchError: Error | null = null;

  // Coba fetch pertama via proxy Next.js /api/predict
  try {
    res = await fetch(`/api${PREDICT_PATH}`, { method: "POST", body: form });
  } catch (e) {
    fetchError = e as Error;
  }

  // Jika proxy gagal (misalnya next dev tanpa rewrite atau direct browser), coba langsung ke API_BASE_URL
  if (!res) {
    try {
      res = await fetch(`${API_BASE_URL}${PREDICT_PATH}`, { method: "POST", body: form });
      fetchError = null;
    } catch (e) {
      fetchError = e as Error;
    }
  }

  if (!res) {
    throw new ApiError(
      0,
      `Backend tidak tersedia (${API_BASE_URL}). Pastikan server FastAPI aktif di port 8000.`
    );
  }

  const body = await res.json().catch(() => null);

  if (!res.ok) {
    throw new ApiError(res.status, errorMessage(body, res.status));
  }

  return normalize(body);
}
