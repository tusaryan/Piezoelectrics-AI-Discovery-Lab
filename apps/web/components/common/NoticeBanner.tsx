"use client";

import { AlertCircle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import type { ReactNode } from "react";

type NoticeVariant = "info" | "success" | "warning" | "error";

interface NoticeBannerProps {
  variant?: NoticeVariant;
  title?: string;
  message: string;
  details?: string[];
  onDismiss?: () => void;
  action?: ReactNode;
  compact?: boolean;
}

const iconByVariant = {
  info: Info,
  success: CheckCircle2,
  warning: AlertCircle,
  error: XCircle,
} as const;

export default function NoticeBanner({
  variant = "info",
  title,
  message,
  details,
  onDismiss,
  action,
  compact = false,
}: NoticeBannerProps) {
  const Icon = iconByVariant[variant];
  return (
    <div className={`notice-banner ${variant}${compact ? " compact" : ""}`} role="alert">
      <div className="notice-icon-wrap">
        <Icon size={16} />
      </div>
      <div className="notice-content">
        {title && <div className="notice-title">{title}</div>}
        <div className="notice-message">{message}</div>
        {!!details?.length && (
          <ul className="notice-details">
            {details.slice(0, 6).map((item, idx) => (
              <li key={`${item}-${idx}`}>{item}</li>
            ))}
          </ul>
        )}
      </div>
      {action && <div className="notice-action">{action}</div>}
      {onDismiss && (
        <button className="notice-dismiss" onClick={onDismiss} aria-label="Dismiss notice">
          <X size={14} />
        </button>
      )}
    </div>
  );
}

