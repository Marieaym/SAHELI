import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { getDistricts, getDistrictHistory } from "../api/client";
import { useLanguage } from "../context/LanguageContext";
import { PageHeader, Card } from "../components/ui";
import RiskBadge from "../components/RiskBadge";

export default function CompareDistricts() {
  const { t } = useLanguage();
  const [districts, setDistricts] = useState(null);
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");
  const [leftHistory, setLeftHistory] = useState(null);
  const [rightHistory, setRightHistory] = useState(null);

  useEffect(() => {
    getDistricts().then((d) => {
      setDistricts(d.districts);
      if (d.districts.length > 1) {
        setLeft(d.districts[0].district);
        setRight(d.districts[1].district);
      }
    });
  }, []);

  useEffect(() => {
    if (left) getDistrictHistory(left, 365).then((d) => setLeftHistory(d.history));
  }, [left]);
  useEffect(() => {
    if (right) getDistrictHistory(right, 365).then((d) => setRightHistory(d.history));
  }, [right]);

  if (!districts) return <div className="text-muted">Loading...</div>;

  const leftData = districts.find((d) => d.district === left);
  const rightData = districts.find((d) => d.district === right);

  // Merge histories on date for overlay chart
  const merged = (leftHistory && rightHistory)
    ? leftHistory.map((h, i) => ({
        date: h.date.slice(0, 10),
        [`${left}_drought`]: h.drought_index,
        [`${right}_drought`]: rightHistory[i]?.drought_index,
      }))
    : [];

  function DistrictCard({ data }) {
    if (!data) return null;
    return (
      <Card>
        <h3 className="font-display text-lg text-sand">{data.district}, {data.country}</h3>
        <span className="text-muted text-xs">{data.zone} {t("compare.zone")}</span>
        <div className="mt-3"><RiskBadge level={data.predicted_risk} /></div>
        <div className="mt-4 space-y-1.5 text-sm">
          <div className="flex justify-between"><span className="text-muted">{t("compare.droughtIndex")}</span><span className="font-mono text-sand">{data.drought_index.toFixed(2)}</span></div>
          <div className="flex justify-between"><span className="text-muted">{t("compare.dryDaysStreak")}</span><span className="font-mono text-sand">{data.consec_dry_days}</span></div>
          <div className="flex justify-between"><span className="text-muted">{t("compare.probCritical")}</span><span className="font-mono text-clay">{(data.prob_critical * 100).toFixed(1)}%</span></div>
        </div>
      </Card>
    );
  }

  return (
    <div>
      <PageHeader eyebrow={t("compare.eyebrow")} title={t("compare.title")} subtitle={t("compare.subtitle")} />

      <div className="grid grid-cols-2 gap-5 mb-6">
        <select value={left} onChange={(e) => setLeft(e.target.value)} className="bg-card border border-cardBorder text-sand rounded-md px-3 py-2">
          {districts.map((d) => <option key={d.district} value={d.district}>{d.district} — {d.country}</option>)}
        </select>
        <select value={right} onChange={(e) => setRight(e.target.value)} className="bg-card border border-cardBorder text-sand rounded-md px-3 py-2">
          {districts.map((d) => <option key={d.district} value={d.district}>{d.district} — {d.country}</option>)}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-5 mb-6">
        <DistrictCard data={leftData} />
        <DistrictCard data={rightData} />
      </div>

      <Card>
        <div className="text-sand font-display text-sm mb-3">Drought Index — 12-Month Overlay</div>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={merged}>
            <XAxis dataKey="date" stroke="#9099B5" fontSize={10} tick={false} />
            <YAxis stroke="#9099B5" fontSize={11} />
            <Tooltip contentStyle={{ backgroundColor: "#1A2238", border: "1px solid #2A3354", borderRadius: 6 }} />
            <Legend />
            <Line type="monotone" dataKey={`${left}_drought`} name={left} stroke="#D6A24A" dot={false} />
            <Line type="monotone" dataKey={`${right}_drought`} name={right} stroke="#6B9080" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
