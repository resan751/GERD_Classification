"use client";

import { useEffect, useRef } from "react";
import type { HistoryRecord } from "@/lib/store";
import AnalysisReportDashboard from "./AnalysisReportDashboard";

type Props = {
  record: HistoryRecord;
  threshold: number;
  onClose: () => void;
  onSave?: () => void;
};

export default function ResultDialog({ record, threshold, onClose, onSave }: Props) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialogEl = ref.current;
    if (!dialogEl) return;

    if (!dialogEl.open) {
      dialogEl.showModal();
    }

    const handleCancel = (e: Event) => {
      e.preventDefault();
      onClose();
    };

    dialogEl.addEventListener("cancel", handleCancel);
    return () => dialogEl.removeEventListener("cancel", handleCancel);
  }, [onClose]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      aria-label="Laporan Analisis Endoskopi"
      className="medical-dialog-wide"
    >
      <div className="medical-dialog-scroll-area">
        <AnalysisReportDashboard
          record={record}
          threshold={threshold}
          isDialog={true}
          onClose={() => {
            ref.current?.close();
            onClose();
          }}
          onSave={onSave}
        />
      </div>
    </dialog>
  );
}
