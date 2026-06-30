import { useState, useRef, useEffect } from "react";
import { Upload, Leaf, AlertTriangle, CheckCircle2, Loader2, Camera, MapPin } from "lucide-react";
import { predictCornLeaf, getDistricts } from "../api/client";
import { PageHeader, Card, SectionLabel } from "../components/ui";
import { useLanguage } from "../context/LanguageContext";

const CLASS_COLOR = {
  "Healthy": "text-acacia",
  "Common Rust": "text-amber",
  "Northern Leaf Blight": "text-clay",
  "Cercospora / Gray Leaf Spot": "text-clay",
};

export default function CornScanner() {
  const { t } = useLanguage();
  const [districts, setDistricts] = useState([]);
  const [district, setDistrict] = useState("");
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    getDistricts().then((d) => setDistricts(d.districts)).catch(() => setDistricts([]));
  }, []);

  async function handleFile(file) {
    if (!file) return;
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(file));
    setLoading(true);
    try {
      const data = await predictCornLeaf(file, district || null);
      setResult(data);
    } catch (e) {
      setError(e?.response?.data?.detail || t("cornScanner.genericError"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("cornScanner.eyebrow")}
        title={t("cornScanner.title")}
        subtitle={t("cornScanner.subtitle")}
      />

      <Card className="mb-4">
        <SectionLabel><MapPin size={11} className="inline -mt-0.5 mr-1" />{t("cornScanner.districtLabel")}</SectionLabel>
        <select value={district} onChange={(e) => setDistrict(e.target.value)} className="input-field w-full sm:w-64">
          <option value="">{t("cornScanner.noDistrict")}</option>
          {districts.map((d) => (
            <option key={d.district} value={d.district}>{d.district}, {d.country}</option>
          ))}
        </select>
        <p className="text-muted text-[11px] mt-2 leading-relaxed">{t("cornScanner.districtHint")}</p>
      </Card>

      <Card className="p-6 md:p-8">
        <div
          className="border-2 border-dashed border-cardBorder rounded-2xl p-8 text-center cursor-pointer hover:border-primary/50 transition-colors"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            handleFile(e.dataTransfer.files?.[0]);
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          {preview ? (
            <img src={preview} alt="Uploaded leaf" className="max-h-64 mx-auto rounded-xl object-contain" />
          ) : (
            <div className="flex flex-col items-center gap-3 text-muted">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center">
                <Camera size={28} className="text-primary" />
              </div>
              <div className="font-display font-semibold text-sand">{t("cornScanner.dropHere")}</div>
              <div className="text-sm">{t("cornScanner.orClick")}</div>
            </div>
          )}
        </div>

        {loading && (
          <div className="flex items-center justify-center gap-2 mt-6 text-muted">
            <Loader2 size={18} className="animate-spin" />
            <span className="font-mono text-sm">{t("cornScanner.running")}</span>
          </div>
        )}

        {error && (
          <div className="mt-6 flex items-center gap-2 text-clay bg-clay/10 rounded-xl p-4">
            <AlertTriangle size={18} />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {result && !loading && (
          <div className="mt-6 animate-fade-up">
            <div className="flex items-center gap-3 bg-surface/80 border border-cardBorder rounded-2xl p-5">
              {result.predicted_class === "Healthy" ? (
                <CheckCircle2 size={28} className="text-acacia flex-shrink-0" />
              ) : (
                <Leaf size={28} className={`flex-shrink-0 ${CLASS_COLOR[result.predicted_class] || "text-amber"}`} />
              )}
              <div className="flex-1">
                <div className={`font-display font-bold text-lg ${CLASS_COLOR[result.predicted_class] || "text-sand"}`}>
                  {result.predicted_class}
                </div>
                <div className="font-mono text-xs text-muted">
                  {t("cornScanner.confidence")}: {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {result.district_logged && (
              <div className="mt-3 flex items-center gap-2 text-primary bg-primary/10 border border-primary/25 rounded-xl px-3 py-2 text-xs">
                <MapPin size={12} className="flex-shrink-0" />
                {t("cornScanner.loggedFor").replace("{district}", result.district_logged)}
              </div>
            )}

            <div className="mt-4 space-y-2">
              {Object.entries(result.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, prob]) => (
                  <div key={cls} className="flex items-center gap-3">
                    <div className="w-44 text-xs text-muted truncate">{cls}</div>
                    <div className="flex-1 h-2 rounded-full bg-cardBorder/40 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary transition-all duration-500"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <div className="w-12 text-right font-mono text-xs text-muted">{(prob * 100).toFixed(1)}%</div>
                  </div>
                ))}
            </div>

            <div className="mt-4 text-[11px] text-muted leading-relaxed border-t border-cardBorder pt-4">
              {result.scope_note}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
