import { RISK_COLORS } from "../api/client";
import { useLanguage } from "../context/LanguageContext";

const LEVEL_FR = { Critical: "Critique", High: "Élevé", Medium: "Moyen", Low: "Faible" };

/* Signature element: risk level reads as a field-stamped tag, like a
   hand-marked index card, not a SaaS status pill. */
export default function RiskBadge({ level, size = "md" }) {
  const { language } = useLanguage();
  const color = RISK_COLORS[level] || "#888888";
  const label = language === "fr" ? (LEVEL_FR[level] || level) : level;
  const sizeClasses = size === "sm" ? "text-[9px] px-1.5 py-0.5" : "text-[10px] px-2 py-1";
  return (
    <span
      className={`inline-flex items-center gap-1 font-mono font-semibold uppercase tracking-wider rounded-[2px] -rotate-1 ${sizeClasses}`}
      style={{
        color,
        border: `1px solid ${color}`,
        backgroundColor: `${color}14`,
      }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}
