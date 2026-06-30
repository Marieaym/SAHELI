import { useEffect, useState } from "react";
import { Smartphone, MessageSquare, Volume2, Share2, FileText, Send, Loader2, CheckCircle2, XCircle } from "lucide-react";
import { getDistricts, getAlert, getSmsStatus, sendAlert } from "../api/client";
import { PageHeader, Card, LoadingState, SectionLabel } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import { useLanguage } from "../context/LanguageContext";

const LANGUAGES = [
  { code: "fr", label: "Français", flag: "🇫🇷" },
  { code: "ha", label: "Hausa", flag: "🇳🇪" },
  { code: "dje", label: "Zarma", flag: "🇳🇪" },
  { code: "wo", label: "Wolof", flag: "🇸🇳" },
  { code: "ar", label: "العربية", flag: "🇲🇷" },
];

export default function AlertSimulator() {
  const { t } = useLanguage();
  const [districts, setDistricts] = useState(null);
  const [selected, setSelected] = useState("");
  const [lang, setLang] = useState("fr");
  const [channel, setChannel] = useState("sms");
  const [alert, setAlert] = useState(null);
  const [smsStatus, setSmsStatus] = useState(null);
  const [phone, setPhone] = useState("");
  const [sending, setSending] = useState(false);
  const [sendResult, setSendResult] = useState(null);

  useEffect(() => {
    getDistricts().then((d) => {
      setDistricts(d.districts);
      if (d.districts.length) setSelected(d.districts[0].district);
    });
    getSmsStatus().then(setSmsStatus).catch(() => setSmsStatus({ sms_configured: false, whatsapp_configured: false }));
  }, []);

  useEffect(() => {
    if (selected) { getAlert(selected, lang).then(setAlert); setSendResult(null); }
  }, [selected, lang]);

  const channelReady = channel === "sms" ? smsStatus?.sms_configured : smsStatus?.whatsapp_configured;

  async function handleSend() {
    if (!phone) return;
    setSending(true);
    setSendResult(null);
    try {
      const res = await sendAlert(selected, phone, lang, channel);
      setSendResult(res);
    } catch (e) {
      setSendResult({ sent: false, error: e?.response?.data?.detail || String(e) });
    } finally {
      setSending(false);
    }
  }

  if (!districts) return <LoadingState message={t("overview.loading")} />;

  const data = districts.find((d) => d.district === selected);

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("alerts.eyebrow")}
        title={t("alerts.title")}
        subtitle={t("alerts.subtitle")}
      />

      <Card className="mb-6">
        <div className="flex gap-6 flex-wrap">
          <div>
            <SectionLabel>{t("alerts.district")}</SectionLabel>
            <select value={selected} onChange={(e) => setSelected(e.target.value)} className="input-field w-56">
              {districts.map((d) => (
                <option key={d.district} value={d.district}>{d.district}</option>
              ))}
            </select>
          </div>
          <div>
            <SectionLabel>{t("alerts.language")}</SectionLabel>
            <div className="flex gap-2">
              {LANGUAGES.map((l) => (
                <button
                  key={l.code}
                  onClick={() => setLang(l.code)}
                  className={`px-4 py-2 rounded-xl text-sm border transition-all ${
                    lang === l.code ? "bg-gold text-night border-gold font-semibold shadow-sm" : "border-cardBorder text-muted hover:text-sand hover:border-gold/40"
                  }`}
                >
                  {l.flag} {l.label}
                </button>
              ))}
            </div>
          </div>
          {data && <div className="flex items-end"><RiskBadge level={data.predicted_risk} /></div>}
        </div>
      </Card>

      {alert && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="flex justify-center lg:justify-start">
            <div className="relative">
              <div className="absolute -inset-4 bg-gold/5 rounded-[40px] blur-xl" />
              <div className="relative bg-card border-[3px] border-cardBorder rounded-[32px] p-5 w-[320px] shadow-xl">
                <div className="bg-night rounded-[24px] p-5 min-h-[260px]">
                  <div className="flex items-center gap-2 text-muted text-[11px] mb-4">
                    <Smartphone size={13} /> {t("alerts.smsNow")}
                  </div>
                  <div
                    dir={lang === "ar" ? "rtl" : "ltr"}
                    className="bg-gold text-night rounded-2xl rounded-tl-sm p-4 text-sm leading-relaxed shadow-md"
                  >
                    {alert.message}
                  </div>
                  <div className="mt-4 flex items-center gap-2 text-muted text-[10px]">
                    <MessageSquare size={11} /> Agent Alerter · {alert.channel}
                  </div>
                </div>
              </div>

              {/* Real send block */}
              <div className="mt-5 bg-card border border-cardBorder rounded-2xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <button
                    onClick={() => setChannel("sms")}
                    className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      channel === "sms" ? "bg-primary text-white" : "bg-night/30 text-muted hover:text-sand"
                    }`}
                  >
                    {t("alerts.channelSms")}
                  </button>
                  <button
                    onClick={() => setChannel("whatsapp")}
                    className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      channel === "whatsapp" ? "bg-primary text-white" : "bg-night/30 text-muted hover:text-sand"
                    }`}
                  >
                    {t("alerts.channelWhatsapp")}
                  </button>
                </div>
                <p className="text-muted text-[10px] mb-3 leading-relaxed">
                  {channel === "sms" ? t("alerts.channelSmsHint") : t("alerts.channelWhatsappHint")}
                </p>
                <div className="flex items-center gap-2 mb-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${channelReady ? "bg-acacia" : "bg-muted"}`} />
                  <span className="text-xs font-mono text-muted">
                    {channelReady ? t("alerts.realSendReady") : t("alerts.realSendNotConfigured")}
                  </span>
                </div>
                <div className="flex gap-2">
                  <input
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    placeholder="+227 90 11 22 33"
                    className="input-field flex-1 text-sm"
                  />
                  <button
                    onClick={handleSend}
                    disabled={sending || !phone}
                    className="btn-primary px-4 flex items-center gap-2 text-sm disabled:opacity-50"
                  >
                    {sending ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                    {t("alerts.sendButton")}
                  </button>
                </div>
                {!channelReady && (
                  <p className="text-muted text-[10px] mt-2 leading-relaxed">
                    {channel === "sms" ? t("alerts.realSendHintSms") : t("alerts.realSendHintWhatsapp")}
                  </p>
                )}
                {sendResult && (
                  <div className={`mt-3 flex items-start gap-2 text-xs rounded-xl p-2.5 ${
                    sendResult.sent ? "bg-acacia/10 text-acacia border border-acacia/30" : "bg-clay/10 text-clay border border-clay/30"
                  }`}>
                    {sendResult.sent ? <CheckCircle2 size={14} className="flex-shrink-0 mt-0.5" /> : <XCircle size={14} className="flex-shrink-0 mt-0.5" />}
                    <span>
                      {sendResult.sent
                        ? t("alerts.sendSuccess").replace("{sid}", sendResult.message_sid)
                        : sendResult.error_code === "no_credentials"
                        ? t("alerts.sendNoCreds")
                        : sendResult.error_code === "recipient_not_verified"
                        ? t("alerts.sendNotVerified")
                        : sendResult.error_code === "recipient_not_in_sandbox"
                        ? t("alerts.sendNotInSandbox")
                        : sendResult.error_code === "daily_limit_reached"
                        ? t("alerts.sendDailyLimit")
                        : t("alerts.sendFailed").replace("{err}", sendResult.error || "")}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-5">
            <Card>
              <div className="flex items-center gap-2 mb-4">
                <span className="text-muted text-sm">{t("alerts.riskTrigger")}</span>
                <RiskBadge level={alert.risk_level} size="sm" />
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-night/40 rounded-xl px-3 py-2.5">
                  <div className="text-muted text-[10px] uppercase">{t("alerts.channel")}</div>
                  <div className="text-sand mt-0.5">{alert.channel}</div>
                </div>
                <div className="bg-night/40 rounded-xl px-3 py-2.5">
                  <div className="text-muted text-[10px] uppercase">{t("alerts.districtLabel")}</div>
                  <div className="text-sand mt-0.5">{alert.district}</div>
                </div>
              </div>
            </Card>

            <Card>
              <SectionLabel>{t("alerts.fullSystemTitle")}</SectionLabel>
              <ul className="space-y-3">
                {[
                  { icon: Volume2, text: t("alerts.voiceNote") },
                  { icon: Share2, text: t("alerts.whatsapp") },
                  { icon: FileText, text: t("alerts.logged") },
                ].map(({ icon: Icon, text }) => (
                  <li key={text} className="flex gap-3 text-sand text-sm">
                    <div className="w-8 h-8 rounded-lg bg-gold/10 flex items-center justify-center flex-shrink-0">
                      <Icon size={14} className="text-gold" />
                    </div>
                    {text}
                  </li>
                ))}
              </ul>
              <p className="text-muted text-[11px] mt-5 border-t border-cardBorder pt-4 leading-relaxed">{alert.disclaimer}</p>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
