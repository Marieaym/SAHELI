import { useEffect, useState } from "react";
import { Download, FileText, CheckCircle2, Satellite, Brain, Globe } from "lucide-react";
import { getDistricts, downloadBrief } from "../api/client";
import { PageHeader, Card, LoadingState, SectionLabel } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import { useLanguage } from "../context/LanguageContext";

const RECOMMENDATIONS = {
  Critical: [
    "Immediate release of emergency food reserves to the district",
    "Deploy mobile health and nutrition screening units within 7 days",
    "Activate cash-transfer program for the most vulnerable households",
    "Pre-position water trucking capacity for pastoral corridors",
  ],
  High: [
    "Place district on elevated monitoring status with weekly re-assessment",
    "Pre-position seed and fodder reserves at regional depots",
    "Issue early advisory to local agricultural extension officers",
  ],
  Medium: ["Maintain standard monitoring cadence", "Verify market price stability through next reporting cycle"],
  Low: ["No intervention required at this time", "Continue routine seasonal monitoring"],
};

export default function PolicyBrief() {
  const { t, language } = useLanguage();
  const [districts, setDistricts] = useState(null);
  const [selected, setSelected] = useState("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    getDistricts().then((d) => {
      setDistricts(d.districts);
      if (d.districts.length) setSelected(d.districts[0].district);
    });
  }, []);

  useEffect(() => {
    setSuccess(false);
    setError(null);
  }, [selected]);

  if (!districts) return <LoadingState message={t("overview.loading")} />;
  const data = districts.find((d) => d.district === selected);

  async function handleGenerate() {
    setGenerating(true);
    setError(null);
    setSuccess(false);
    try {
      await downloadBrief(selected, language);
      setSuccess(true);
    } catch {
      setError(t("brief.error"));
    } finally {
      setGenerating(false);
    }
  }

  const pdfItems = [t("brief.pdfItem1"), t("brief.pdfItem2"), t("brief.pdfItem3"), t("brief.pdfItem4"), t("brief.pdfItem5")];

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("brief.eyebrow")}
        title={t("brief.title")}
        subtitle={t("brief.subtitle")}
      />

      <Card className="mb-6">
        <SectionLabel>{t("brief.selectDistrict")}</SectionLabel>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="input-field max-w-md"
        >
          {districts.map((d) => (
            <option key={d.district} value={d.district}>
              {d.district} — {d.country}
            </option>
          ))}
        </select>
      </Card>

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <Card className="lg:col-span-2">
            <div className="flex flex-wrap items-start justify-between gap-4 mb-5">
              <div>
                <h3 className="font-display text-xl text-sand">{data.district}, {data.country}</h3>
                <span className="text-muted text-xs">{data.zone} zone</span>
              </div>
              <RiskBadge level={data.predicted_risk} />
            </div>

            <SectionLabel>{t("brief.indicators")}</SectionLabel>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              {[
                { label: t("brief.droughtIndex"), value: data.drought_index.toFixed(2) },
                { label: t("brief.dryDays"), value: data.consec_dry_days },
                { label: t("brief.ndvi"), value: (data.sentinel2_ndvi ?? 0).toFixed(3) },
                { label: t("brief.conflict"), value: Math.round(data.conflict_events_30d ?? 0) },
              ].map(({ label, value }) => (
                <div key={label} className="bg-night/50 rounded-xl px-3 py-3 border border-cardBorder">
                  <div className="text-muted text-[10px] uppercase tracking-wide">{label}</div>
                  <div className="font-mono text-lg text-sand mt-1">{value}</div>
                </div>
              ))}
            </div>

            {error && (
              <div className="mb-4 text-clay text-sm bg-clay/10 border border-clay/30 rounded-xl px-4 py-3">{error}</div>
            )}
            {success && (
              <div className="mb-4 flex items-center gap-2 text-acacia text-sm bg-acacia/10 border border-acacia/30 rounded-xl px-4 py-3">
                <CheckCircle2 size={16} /> PDF generated and downloaded successfully.
              </div>
            )}

            <button
              onClick={handleGenerate}
              disabled={generating}
              className="btn-primary w-full md:w-auto"
            >
              {generating ? (
                <><FileText size={16} className="animate-pulse" /> {t("brief.generating")}</>
              ) : (
                <><Download size={16} /> {t("brief.generate")}</>
              )}
            </button>
          </Card>

          <div className="space-y-5">
            <Card>
              <SectionLabel>{t("brief.pdfContents")}</SectionLabel>
              <ul className="space-y-2.5">
                {pdfItems.map((item, i) => {
                  const icons = [FileText, Satellite, Globe, Brain, CheckCircle2];
                  const Icon = icons[i] || FileText;
                  return (
                    <li key={item} className="flex gap-2.5 text-sand text-sm">
                      <Icon size={15} className="text-gold flex-shrink-0 mt-0.5" />
                      {item}
                    </li>
                  );
                })}
              </ul>
            </Card>

            <Card>
              <SectionLabel>{t("brief.previewActions")}</SectionLabel>
              <ul className="space-y-2">
                {(RECOMMENDATIONS[data.predicted_risk] || []).map((r) => (
                  <li key={r} className="text-sand text-sm flex gap-2">
                    <span className="text-gold">☐</span> {r}
                  </li>
                ))}
              </ul>
              <p className="text-muted text-xs mt-4 leading-relaxed">{t("brief.previewNote")}</p>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
