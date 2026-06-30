import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { MapContainer, TileLayer, CircleMarker, Popup, Polygon, useMap } from "react-leaflet";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from "recharts";
import { getDistricts, getDistrictHistory } from "../api/client";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import { PageHeader, Card, Metric } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import "leaflet/dist/leaflet.css";

const RISK_COLORS = { Low: "#6B9080", Medium: "#B89B4A", High: "#D9822B", Critical: "#B83227" };
const ZONE_COLORS = { Saharan: "#D9822B", Sahelian: "#D6A24A", Sudanian: "#6B9080", Guinean: "#5B8AA6" };

const TILE_LAYERS = {
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics",
  },
  street: {
    url: `https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png`,
    attribution: "&copy; OpenStreetMap &copy; CARTO",
  },
};

// Andrew's monotone chain convex hull — small point sets, no extra dependency needed.
function convexHull(points) {
  if (points.length < 3) return points;
  const pts = [...points].sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop(); lower.pop();
  return [...lower, ...upper];
}

// Pad a hull outward slightly so markers sit inside the zone outline, not exactly on its edge.
function padHull(hull, factor = 0.35) {
  const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length;
  const cy = hull.reduce((s, p) => s + p[1], 0) / hull.length;
  return hull.map(([x, y]) => [x + (x - cx) * factor, y + (y - cy) * factor]);
}

function FlyToDistrict({ district, districts }) {
  const map = useMap();
  useEffect(() => {
    if (!district) return;
    const d = districts.find((x) => x.district === district);
    if (d) map.flyTo([d.lat, d.lon], 13, { duration: 1.2 });
  }, [district, districts, map]);
  return null;
}

export default function RiskMap() {
  const { t } = useLanguage();
  const { theme } = useTheme();
  const [searchParams] = useSearchParams();
  const districtParam = searchParams.get("district");

  const [districts, setDistricts] = useState(null);
  const [selected, setSelected] = useState(null);
  const [history, setHistory] = useState(null);
  const [showZones, setShowZones] = useState(true);
  const [mapLayer, setMapLayer] = useState("satellite");

  useEffect(() => {
    getDistricts().then((d) => {
      setDistricts(d.districts);
      setSelected(districtParam || d.districts[0].district);
    });
  }, [districtParam]);

  useEffect(() => {
    if (selected) {
      getDistrictHistory(selected, 365).then((d) => setHistory(d.history));
    }
  }, [selected]);

  if (!districts) return <div className="text-muted">Loading districts...</div>;

  const nCritical = districts.filter((d) => d.predicted_risk === "Critical").length;
  const nHigh = districts.filter((d) => d.predicted_risk === "High").length;
  const avgDrought = (districts.reduce((s, d) => s + d.drought_index, 0) / districts.length).toFixed(2);
  const selectedData = districts.find((d) => d.district === selected);
  const sorted = [...districts].sort((a, b) => {
    const order = { Critical: 0, High: 1, Medium: 2, Low: 3 };
    return order[a.predicted_risk] - order[b.predicted_risk];
  });

  const zoneHulls = Object.keys(ZONE_COLORS)
    .map((zone) => {
      const pts = districts.filter((d) => d.zone === zone).map((d) => [d.lat, d.lon]);
      if (pts.length < 3) return null;
      return { zone, hull: padHull(convexHull(pts)) };
    })
    .filter(Boolean);

  const chartHistory = (history || []).map((h) => ({
    date: h.date.slice(0, 10),
    precip_30d: h.precip_30d,
    drought_index: h.drought_index,
  }));

  const probData = selectedData
    ? [
        { name: "Low", value: selectedData.prob_low },
        { name: "Medium", value: selectedData.prob_medium },
        { name: "High", value: selectedData.prob_high },
        { name: "Critical", value: selectedData.prob_critical },
      ]
    : [];

  return (
    <div>
      <PageHeader eyebrow={t("map.eyebrow")} title={t("map.title")} />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Metric label={t("map.districtsMonitored")} value={districts.length} />
        <Metric label={t("map.criticalRisk")} value={nCritical} sublabel={`${nCritical}/${districts.length}`} />
        <Metric label={t("map.highRisk")} value={nHigh} sublabel={`${nHigh}/${districts.length}`} />
        <Metric label={t("map.avgDrought")} value={avgDrought} sublabel="lower = drier" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-6">
        <Card className="lg:col-span-2 p-0 overflow-hidden relative">
          <button
            onClick={() => setShowZones((v) => !v)}
            className={`absolute top-3 right-3 z-[1000] text-xs px-3 py-1.5 rounded-md border font-mono ${
              showZones ? "bg-gold text-night border-gold" : "bg-night text-muted border-cardBorder"
            }`}
          >
            {t("map.zoneOverlays")} {showZones ? "On" : "Off"}
          </button>
          <div className="absolute top-3 left-3 z-[1000] flex rounded-md border border-cardBorder overflow-hidden font-mono text-xs">
            <button
              onClick={() => setMapLayer("satellite")}
              className={`px-3 py-1.5 ${mapLayer === "satellite" ? "bg-gold text-night" : "bg-night text-muted"}`}
            >
              Satellite
            </button>
            <button
              onClick={() => setMapLayer("street")}
              className={`px-3 py-1.5 ${mapLayer === "street" ? "bg-gold text-night" : "bg-night text-muted"}`}
            >
              Map
            </button>
          </div>
          <MapContainer center={[14.5, 2.0]} zoom={4.3} maxZoom={18} style={{ height: "480px", width: "100%", backgroundColor: "#12182B" }}>
            <TileLayer
              key={mapLayer}
              url={TILE_LAYERS[mapLayer].url}
              attribution={TILE_LAYERS[mapLayer].attribution}
              maxZoom={18}
            />
            <FlyToDistrict district={selected} districts={districts} />

            {showZones && zoneHulls.map(({ zone, hull }) => (
              <Polygon
                key={zone}
                positions={hull}
                pathOptions={{ color: ZONE_COLORS[zone], weight: 1.5, fillColor: ZONE_COLORS[zone], fillOpacity: 0.06, dashArray: "4 4" }}
              />
            ))}

            {/* Pulsing halo behind Critical-risk markers */}
            {districts.filter((d) => d.predicted_risk === "Critical").map((d) => (
              <CircleMarker
                key={`pulse-${d.district}`}
                center={[d.lat, d.lon]}
                radius={20}
                pathOptions={{ color: RISK_COLORS.Critical, fillColor: RISK_COLORS.Critical, fillOpacity: 0.35, weight: 0, className: "pulse-critical" }}
                interactive={false}
              />
            ))}

            {districts.map((d) => (
              <CircleMarker
                key={d.district}
                center={[d.lat, d.lon]}
                radius={d.district === selected ? 15 : d.predicted_risk === "Critical" || d.predicted_risk === "High" ? 13 : 10}
                pathOptions={{
                  color: d.district === selected ? "#F2EDE3" : RISK_COLORS[d.predicted_risk],
                  fillColor: RISK_COLORS[d.predicted_risk],
                  fillOpacity: 0.75,
                  weight: d.district === selected ? 3 : 2,
                }}
                eventHandlers={{ click: () => setSelected(d.district) }}
              >
                <Popup>
                  <b>{d.district}</b> ({d.country})<br />
                  Zone: {d.zone}<br />
                  Risk: <b style={{ color: RISK_COLORS[d.predicted_risk] }}>{d.predicted_risk}</b><br />
                  Drought Index: {d.drought_index.toFixed(2)}
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
          {showZones && (
            <div className="absolute bottom-3 left-3 z-[1000] flex gap-3 bg-night/80 rounded-md px-3 py-1.5 text-[10px]">
              {Object.entries(ZONE_COLORS).map(([zone, color]) => (
                <span key={zone} className="flex items-center gap-1 text-muted">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                  {zone}
                </span>
              ))}
            </div>
          )}
        </Card>

        <Card className="p-0 overflow-hidden">
          <div className="px-4 pt-4 pb-2 font-display text-sm text-sand">{t("map.riskRanking")}</div>
          <div className="overflow-y-auto" style={{ maxHeight: 440 }}>
            <table className="w-full text-sm">
              <tbody>
                {sorted.map((d) => (
                  <tr
                    key={d.district}
                    onClick={() => setSelected(d.district)}
                    className={`cursor-pointer border-t border-cardBorder hover:bg-white/5 ${
                      selected === d.district ? "bg-gold/10" : ""
                    }`}
                  >
                    <td className="px-4 py-2">
                      <div className="text-sand">{d.district}</div>
                      <div className="text-muted text-xs">{d.country} · {d.zone}</div>
                    </td>
                    <td className="px-4 py-2 text-right">
                      <RiskBadge level={d.predicted_risk} size="sm" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {selectedData && (
        <Card>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-display text-lg text-sand">
                {selectedData.district}, {selectedData.country}
              </h3>
              <span className="text-muted text-xs">{selectedData.zone} zone</span>
            </div>
            <RiskBadge level={selectedData.predicted_risk} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="text-muted text-xs mb-2">30-day rainfall & drought index trend</div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartHistory}>
                  <XAxis dataKey="date" stroke="#9099B5" fontSize={10} tick={false} />
                  <YAxis stroke="#9099B5" fontSize={11} />
                  <Tooltip contentStyle={{ backgroundColor: "#1A2238", border: "1px solid #2A3354", borderRadius: 6 }} />
                  <Line type="monotone" dataKey="precip_30d" stroke="#6B9080" name="30d Rainfall (mm)" dot={false} />
                  <Line type="monotone" dataKey="drought_index" stroke="#B83227" name={t("map.droughtIndexLine")} dot={false} strokeDasharray="4 2" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div>
              <div className="text-muted text-xs mb-2">{t("map.riskProbDist")}</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={probData} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" stroke="#9099B5" fontSize={10} />
                  <YAxis type="category" dataKey="name" stroke="#9099B5" fontSize={11} width={60} />
                  <Tooltip contentStyle={{ backgroundColor: "#1A2238", border: "1px solid #2A3354", borderRadius: 6 }} />
                  <Bar dataKey="value">
                    {probData.map((entry) => (
                      <Cell key={entry.name} fill={RISK_COLORS[entry.name]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              <div className="mt-3 bg-night border border-cardBorder rounded p-3 text-xs text-muted">
                <b className="text-sand">{t("map.whyPrediction")}</b>
                <ul className="list-disc list-inside mt-1 space-y-0.5">
                  {selectedData.drought_index < -0.5 && (
                    <li>{t("map.waterBalanceDriver", { v: Math.abs(selectedData.drought_index).toFixed(1) })}</li>
                  )}
                  {selectedData.consec_dry_days >= 15 && <li>{t("map.dryDaysDriver", { d: selectedData.consec_dry_days })}</li>}
                  {selectedData.drought_index >= -0.5 && selectedData.consec_dry_days < 15 && (
                    <li>{t("map.normalRange")}</li>
                  )}
                </ul>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-cardBorder grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-muted text-[10px] uppercase tracking-wide">Pastoral water access (real OSM)</div>
              <div className="font-display text-xl text-sand">
                {Math.round(selectedData.water_point_count_50km ?? 0)} points
              </div>
              <div className="text-muted text-[11px]">mapped within 50km — coverage varies by OSM survey density</div>
            </div>
            <div>
              <div className="text-muted text-[10px] uppercase tracking-wide">Groundwater anomaly (real GRACE-FO)</div>
              <div className="font-display text-xl text-sand">
                {(selectedData.groundwater_anomaly_cm ?? 0).toFixed(1)} cm
              </div>
              <div className="text-muted text-[11px]">liquid water equivalent, vs. 2004-2009 baseline</div>
            </div>
            <div>
              <div className="text-muted text-[10px] uppercase tracking-wide">Vegetation health (real Sentinel-2)</div>
              <div className="font-display text-xl text-sand">
                {(selectedData.sentinel2_ndvi ?? 0).toFixed(3)} NDVI
              </div>
              <div className="text-muted text-[11px]">
                {selectedData.sentinel2_scene_date
                  ? `scene ${selectedData.sentinel2_scene_date.slice(0, 10)}, ${selectedData.sentinel2_cloud_cover_pct?.toFixed(1)}% cloud`
                  : t("map.sceneUnavailable")}
              </div>
            </div>
            {selectedData.ipc_phase_observed != null && (
              <div>
                <div className="text-muted text-[10px] uppercase tracking-wide">{t("map.realIpcPhase")}</div>
                <div className="font-display text-xl text-sand">{selectedData.ipc_phase_observed.toFixed(1)} / 5</div>
                <div className="text-muted text-[11px]">{t("map.groundTruthRecent")}</div>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
