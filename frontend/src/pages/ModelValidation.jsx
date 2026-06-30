import { useEffect, useState } from "react";
import { AlertCircle, CheckCircle2 } from "lucide-react";
import { getModelValidation, getFederatedResults, getRLResults, getEdgeResults, getGroundTruthValidation, getMonsoonSignal } from "../api/client";
import { PageHeader, Card, Metric, LoadingState } from "../components/ui";
import { useLanguage } from "../context/LanguageContext";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";

export default function ModelValidation() {
  const { t } = useLanguage();
  const [data, setData] = useState(null);
  const [federated, setFederated] = useState(null);
  const [rl, setRl] = useState(null);
  const [edge, setEdge] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);
  const [monsoon, setMonsoon] = useState(null);

  useEffect(() => {
    getModelValidation().then(setData);
    getFederatedResults().then(setFederated).catch(() => setFederated(null));
    getRLResults().then(setRl).catch(() => setRl(null));
    getEdgeResults().then(setEdge).catch(() => setEdge(null));
    getGroundTruthValidation().then(setGroundTruth).catch(() => setGroundTruth(null));
    getMonsoonSignal().then(setMonsoon).catch(() => setMonsoon(null));
  }, []);

  if (!data) return <LoadingState message={t("validation.loading")} />;

  const perClassData = Object.entries(data.classification_report)
    .filter(([k]) => ["Low", "Medium", "High", "Critical"].includes(k))
    .map(([level, m]) => ({
      level,
      precision: +(m.precision * 100).toFixed(1),
      recall: +(m.recall * 100).toFixed(1),
      f1: +(m["f1-score"] * 100).toFixed(1),
      support: m.support,
    }));

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("validation.eyebrow")}
        title={t("validation.title")}
        subtitle={t("validation.subtitle")}
      />

      <div className="flex items-start gap-3 bg-amber/10 border border-amber/30 rounded-xl px-4 py-3 mb-6">
        <AlertCircle size={18} className="text-amber flex-shrink-0 mt-0.5" />
        <p className="text-sand text-sm leading-relaxed">{data.validation_note}</p>
      </div>

      {groundTruth && (
        <Card className="mb-6 border-2 border-clay/40">
          <div className="flex items-center justify-between mb-3">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-clay">
              Real Ground-Truth Check — vs. FEWS NET IPC Classification
            </span>
          </div>
          <div className="flex flex-wrap items-baseline gap-3 mb-3">
            <span className="font-display font-semibold text-3xl text-clay">
              ρ = {groundTruth.spearman_rho}
            </span>
            <span className="text-muted text-sm">
              correlation vs. real, independent FEWS NET classification ({groundTruth.n_observations.toLocaleString()} observations, {groundTruth.n_districts_matched} districts)
            </span>
          </div>
          <p className="text-sand text-sm leading-relaxed mb-3">{groundTruth.honest_interpretation}</p>
          <details className="text-xs text-muted" open>
            <summary className="cursor-pointer text-gold">{t("validation.perDistrictDetail")}</summary>
            <div className="mt-2 flex flex-wrap gap-2">
              {groundTruth.per_district_rho?.map((d) => (
                <span key={d.district} className="font-mono text-[11px] px-2 py-1 rounded-[2px] border border-cardBorder">
                  {d.district}: ρ={d.rho}
                </span>
              ))}
            </div>
            <ul className="list-disc list-inside mt-3 space-y-1">
              {groundTruth.honest_limitations?.map((l, i) => <li key={i}>{l}</li>)}
            </ul>
          </details>
        </Card>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Metric label={t("validation.testAccuracy")} value={`${(data.accuracy * 100).toFixed(1)}%`} accent="success" />
        <Metric label={t("validation.weightedF1")} value={`${(data.weighted_f1 * 100).toFixed(1)}%`} accent="success" />
        <Metric label={t("validation.trainingRows")} value={data.n_train.toLocaleString()} />
        <Metric label={t("validation.testRows")} value={data.n_test.toLocaleString()} />
      </div>

      {data.validation_methodology && (
        <Card className="mb-6">
          <div className="text-sand font-display text-sm mb-2">{t("validation.methodologyTitle")}</div>
          <p className="text-muted text-sm leading-relaxed">{data.validation_methodology.current}</p>
          <p className="text-muted text-xs leading-relaxed mt-2 border-t border-cardBorder pt-3">
            {t("validation.oldSplitNote")
              .replace("{old}", (data.validation_methodology.old_random_split_accuracy_for_comparison * 100).toFixed(2))
              .replace("{new}", (data.accuracy * 100).toFixed(2))}
          </p>
          <p className="text-muted text-xs leading-relaxed mt-2">{data.validation_methodology.why_changed}</p>
        </Card>
      )}

      {data.linear_baseline_comparison && (
        <Card className="mb-6">
          <div className="text-sand font-display text-sm mb-3">{t("validation.linearBaselineTitle")}</div>
          <div className="grid grid-cols-2 gap-4 mb-3">
            <div className="bg-surface/60 rounded-xl p-3 border border-cardBorder">
              <div className="text-muted text-[10px] uppercase font-mono">XGBoost (SAHELI)</div>
              <div className="font-display font-bold text-2xl text-acacia mt-1">
                {(data.linear_baseline_comparison.xgboost_accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-surface/60 rounded-xl p-3 border border-cardBorder">
              <div className="text-muted text-[10px] uppercase font-mono">{t("validation.linearModel")}</div>
              <div className="font-display font-bold text-2xl text-sand mt-1">
                {(data.linear_baseline_comparison.linear_accuracy * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <p className="text-muted text-xs leading-relaxed">{data.linear_baseline_comparison.interpretation}</p>
        </Card>
      )}

      <Card className="mb-6">
        <div className="text-sand font-display text-sm mb-3">{t("validation.perClassChart")}</div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={perClassData}>
            <XAxis dataKey="level" stroke="rgb(var(--c-muted))" fontSize={12} />
            <YAxis stroke="rgb(var(--c-muted))" fontSize={12} domain={[0, 100]} />
            <Tooltip contentStyle={{ backgroundColor: "rgb(var(--c-card))", border: "1px solid rgb(var(--c-cardBorder))", borderRadius: 6 }} />
            <Legend />
            <Bar dataKey="precision" fill="#D6A24A" />
            <Bar dataKey="recall" fill="#6B9080" />
            <Bar dataKey="f1" fill="#D9822B" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-0 overflow-hidden mb-6">
        <div className="px-5 pt-4 pb-2 font-display text-sm text-sand">{t("validation.perClassDetail")}</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-muted text-xs uppercase border-t border-cardBorder">
              <th className="text-left px-5 py-2">{t("validation.riskLevel")}</th>
              <th className="text-right px-3 py-2">{t("validation.precision")}</th>
              <th className="text-right px-3 py-2">{t("validation.recall")}</th>
              <th className="text-right px-3 py-2">{t("validation.f1")}</th>
              <th className="text-right px-5 py-2">{t("validation.testSamples")}</th>
            </tr>
          </thead>
          <tbody>
            {perClassData.map((row) => (
              <tr key={row.level} className="border-t border-cardBorder">
                <td className="px-5 py-2 text-sand">{row.level}</td>
                <td className="px-3 py-2 text-right font-mono text-muted">{row.precision}%</td>
                <td className="px-3 py-2 text-right font-mono text-muted">{row.recall}%</td>
                <td className="px-3 py-2 text-right font-mono text-muted">{row.f1}%</td>
                <td className="px-5 py-2 text-right font-mono text-muted">{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card className="p-0 overflow-hidden">
        <div className="px-5 pt-4 pb-2 font-display text-sm text-sand">{t("validation.confusionMatrix")}</div>
        <div className="p-5 overflow-x-auto">
          <table className="text-sm font-mono">
            <thead>
              <tr>
                <th className="px-3 py-1 text-muted text-xs">{t("validation.actualPredicted")}</th>
                {data.confusion_matrix_labels.map((l) => (
                  <th key={l} className="px-3 py-1 text-muted text-xs">{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.confusion_matrix.map((row, i) => (
                <tr key={i}>
                  <td className="px-3 py-1 text-sand">{data.confusion_matrix_labels[i]}</td>
                  {row.map((val, j) => (
                    <td
                      key={j}
                      className={`px-3 py-1 text-center ${i === j ? "text-acacia font-semibold" : "text-muted"}`}
                    >
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {federated && (
        <Card className="mb-6">
          <div className="flex items-center justify-between mb-3">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-gold">
              Federated Learning — Flower, FedAvg + Differential Privacy
            </span>
          </div>
          <div className="flex flex-wrap gap-6 mb-4">
            <Metric
              label={t("validation.fedDpAccuracy")}
              value={`${(federated.final_federated_dp_accuracy * 100).toFixed(1)}%`}
              sublabel={`6 country-clients, ${federated.setup.n_rounds} rounds`}
            />
            <Metric
              label={t("validation.centralizedBaseline")}
              value={`${(federated.centralized_baseline_accuracy * 100).toFixed(1)}%`}
              sublabel={t("validation.centralizedSublabel")}
            />
            <Metric
              label={t("validation.privacyCost")}
              value={`-${(federated.privacy_utility_cost * 100).toFixed(1)} pts`}
              sublabel={`ε = ${federated.setup.differential_privacy.epsilon} (Laplace)`}
            />
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={federated.round_history}>
              <XAxis dataKey="round" stroke="rgb(var(--c-muted))" fontSize={11} />
              <YAxis stroke="rgb(var(--c-muted))" fontSize={11} domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: "rgb(var(--c-card))", border: "1px solid rgb(var(--c-cardBorder))", borderRadius: 6 }} />
              <Line type="monotone" dataKey="mean_test_accuracy" stroke="rgb(var(--c-gold))" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-muted text-xs mt-3">
            Each country trains only on its own real climate data; raw data never pooled.
            Local simulation (no Ray-distributed runtime) — see full disclosure in the project README.
          </p>
        </Card>
      )}

      {monsoon && (
        <Card className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-gold">
              Seasonal Forecast Signal — Real NOAA ERSST, 6 Ocean Basins
            </span>
          </div>
          <p className="text-sand text-sm leading-relaxed mb-3">{monsoon.honest_interpretation}</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(monsoon.basin_results || {}).map(([basin, r]) => (
              <span
                key={basin}
                className="inline-flex items-center gap-1.5 text-[11px] font-mono px-2 py-1 rounded-[2px] border"
                style={{
                  color: r.significant_at_0_05 || r["significant_at_0.05"] ? "#5A6E4C" : "#6E6353",
                  borderColor: r.significant_at_0_05 || r["significant_at_0.05"] ? "#5A6E4C" : "#D6C8AD",
                  backgroundColor: r.significant_at_0_05 || r["significant_at_0.05"] ? "#5A6E4C14" : "transparent",
                }}
              >
                {basin.replace(/_/g, " ")}: r={r.pearson_r} (p={r.p_value})
              </span>
            ))}
          </div>
          <details className="text-xs text-muted mt-3">
            <summary className="cursor-pointer text-gold">{t("validation.honestLimitations")}</summary>
            <ul className="list-disc list-inside mt-2 space-y-1">
              {monsoon.honest_limitations?.map((l, i) => <li key={i}>{l}</li>)}
            </ul>
          </details>
        </Card>
      )}

      {rl && (
        <Card>
          <div className="flex items-center justify-between mb-3">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-gold">
              Intervention Agent — PPO, Stable-Baselines3
            </span>
          </div>
          <div className="flex flex-wrap items-baseline gap-3 mb-3">
            <span className="font-display font-semibold text-2xl text-acacia inline-flex items-center gap-2">
              <CheckCircle2 size={20} />+{rl.improvement_over_heuristic_pct}%
            </span>
            <span className="text-muted text-sm">
              more risk resolved than the proportional-allocation heuristic, on the same
              real district severity distribution and the same response dynamics
            </span>
          </div>
          <div className="flex flex-wrap gap-6">
            <Metric label={t("validation.ppoPolicy")} value={rl.mean_severity_points_resolved.ppo_policy} sublabel={t("validation.meanSeverityResolved")} />
            <Metric label={t("validation.heuristic")} value={rl.mean_severity_points_resolved.proportional_heuristic} sublabel={t("validation.meanSeverityResolved")} />
            <Metric label={t("validation.training")} value={rl.setup.training_timesteps.toLocaleString()} sublabel={t("validation.timesteps")} />
          </div>
        </Card>
      )}

      {edge && (
        <Card className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-gold">
              Edge Readiness — Real ONNX Export
            </span>
          </div>
          <div className="flex flex-wrap gap-6 mb-3">
            <Metric label={t("validation.onnxFileSize")} value={`${edge.onnx_file_size_kb} KB`} sublabel={t("validation.quantizationReady")} />
            <Metric label={t("validation.predictionAgreement")} value={`${(edge.prediction_agreement_vs_original * 100).toFixed(0)}%`} sublabel={t("validation.vsOriginal500")} />
            <Metric label={t("validation.cpuInference")} value={`${edge.cpu_inference_latency_ms_per_request} ms`} sublabel={t("validation.perRequest")} />
            <Metric label={t("validation.ramFootprint")} value={`${edge.model_footprint_vs_budget_pct}%`} sublabel={t("validation.piBudget")} />
          </div>
          <details className="text-xs text-muted">
            <summary className="cursor-pointer text-gold">{t("validation.honestLimitationsBenchmark")}</summary>
            <ul className="list-disc list-inside mt-2 space-y-1">
              {edge.honest_limitations?.map((l, i) => <li key={i}>{l}</li>)}
            </ul>
          </details>
        </Card>
      )}
    </div>
  );
}
