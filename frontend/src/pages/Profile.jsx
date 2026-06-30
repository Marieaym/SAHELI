import { useEffect, useRef, useState } from "react";
import { Camera, Save, Sun, Moon, Loader2 } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import { updateProfile, getMyActivity } from "../api/client";
import { PageHeader, Card } from "../components/ui";

const MAX_PHOTO_BYTES = 1_500_000;

function fileToCompressedDataUrl(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const reader = new FileReader();
    reader.onload = () => {
      img.onload = () => {
        const size = 256; // square avatar — small enough to store cheaply in SQLite
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");
        const scale = Math.max(size / img.width, size / img.height);
        const w = img.width * scale, h = img.height * scale;
        ctx.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);
        resolve(canvas.toDataURL("image/jpeg", 0.85));
      };
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function ActivityHeatmap({ counts, language }) {
  const today = new Date();
  const days = [];
  for (let i = 364; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    days.push({ date: d, key, count: counts[key] || 0 });
  }
  // pad to start on a Sunday so weeks line up into clean columns
  const lead = days[0].date.getDay();
  const padded = Array.from({ length: lead }, () => null).concat(days);
  const weeks = [];
  for (let i = 0; i < padded.length; i += 7) weeks.push(padded.slice(i, i + 7));

  function shade(count) {
    if (!count) return "rgb(var(--c-cardBorder))";
    if (count === 1) return "#C9A86A";
    if (count <= 3) return "#A86E2A";
    if (count <= 6) return "#7A5220";
    return "#5A6E4C";
  }

  const monthLabels = language === "fr"
    ? ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    : ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  let lastMonth = -1;
  return (
    <div className="overflow-x-auto pb-2">
      <div className="inline-flex gap-[3px]">
        {weeks.map((week, wi) => {
          const firstReal = week.find((d) => d);
          const month = firstReal ? firstReal.date.getMonth() : -1;
          const showLabel = month !== lastMonth && firstReal && firstReal.date.getDate() <= 7;
          if (showLabel) lastMonth = month;
          return (
            <div key={wi} className="flex flex-col gap-[3px] relative">
              {showLabel && (
                <span className="absolute -top-5 text-[9px] text-muted whitespace-nowrap">{monthLabels[month]}</span>
              )}
              {week.map((d, di) => (
                <div
                  key={di}
                  title={d ? `${d.key}: ${d.count} ${language === "fr" ? "action(s)" : "action(s)"}` : ""}
                  className="w-[11px] h-[11px] rounded-[2px]"
                  style={{ backgroundColor: d ? shade(d.count) : "transparent" }}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function Profile() {
  const { user, updateUser } = useAuth();
  const { language, setLanguage, t } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const fileInputRef = useRef(null);

  const [fullName, setFullName] = useState(user?.full_name || "");
  const [organization, setOrganization] = useState(user?.organization || "");
  const [bio, setBio] = useState(user?.bio || "");
  const [photo, setPhoto] = useState(user?.photo_base64 || null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [activity, setActivity] = useState(null);

  useEffect(() => {
    getMyActivity().then(setActivity).catch(() => setActivity(null));
  }, []);

  async function handlePhotoChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError(language === "fr" ? "Merci de choisir un fichier image." : "Please choose an image file.");
      return;
    }
    try {
      const dataUrl = await fileToCompressedDataUrl(file);
      if (dataUrl.length > MAX_PHOTO_BYTES) {
        setError(language === "fr" ? "Image trop volumineuse après compression." : "Image too large after compression.");
        return;
      }
      setPhoto(dataUrl);
      setError("");
    } catch {
      setError(language === "fr" ? "Impossible de lire cette image." : "Could not read this image.");
    }
  }

  async function handleSave() {
    setSaving(true);
    setError("");
    try {
      const updated = await updateProfile({
        full_name: fullName, organization, bio,
        ...(photo !== user?.photo_base64 ? { photo_base64: photo } : {}),
      });
      updateUser(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (err) {
      setError(err?.response?.data?.detail || (language === "fr" ? "Échec de l'enregistrement." : "Save failed."));
    } finally {
      setSaving(false);
    }
  }

  const initials = (user?.full_name || "?").split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase();

  return (
    <div>
      <PageHeader
        eyebrow={language === "fr" ? "Compte" : "Account"}
        title={language === "fr" ? "Profil" : "Profile"}
        subtitle={language === "fr"
          ? "Gère ta photo, tes informations, tes préférences, et consulte ton activité réelle sur SAHELI."
          : "Manage your photo, profile details, preferences, and your real SAHELI activity history."}
      />

      <div className="grid lg:grid-cols-3 gap-5">
        {/* Photo + identity card */}
        <Card className="lg:col-span-1">
          <div className="flex flex-col items-center text-center">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="relative group w-24 h-24 rounded-full overflow-hidden border-2 border-cardBorder mb-3"
              title={language === "fr" ? "Changer la photo" : "Change photo"}
            >
              {photo ? (
                <img src={photo} alt="avatar" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full bg-gradient-to-br from-gold to-amber flex items-center justify-center text-night text-2xl font-bold">
                  {initials}
                </div>
              )}
              <div className="absolute inset-0 bg-night/0 group-hover:bg-night/50 flex items-center justify-center transition-colors">
                <Camera size={20} className="text-white opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </button>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handlePhotoChange} className="hidden" />
            <div className="font-display font-semibold text-lg text-sand">{user?.full_name}</div>
            <div className="text-muted text-sm">{user?.email}</div>
            <div className="text-muted text-xs mt-1">{user?.country}</div>
          </div>

          <div className="mt-5 pt-4 border-t border-cardBorder">
            <div className="text-[10px] uppercase tracking-widest text-muted font-mono mb-2">
              {language === "fr" ? "Préférences" : "Preferences"}
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-sand">{language === "fr" ? "Langue" : "Language"}</span>
              <div className="flex rounded-lg overflow-hidden border border-cardBorder text-[10px] font-mono">
                {["en", "fr"].map((lng) => (
                  <button key={lng} onClick={() => setLanguage(lng)}
                    className={`px-2.5 py-1 ${language === lng ? "bg-primary text-night font-semibold" : "text-muted"}`}>
                    {lng.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-sand">{language === "fr" ? "Thème" : "Theme"}</span>
              <button onClick={toggleTheme} className="btn-ghost p-1.5 flex items-center gap-1.5 text-xs">
                {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
                {theme === "dark" ? (language === "fr" ? "Clair" : "Light") : (language === "fr" ? "Sombre" : "Dark")}
              </button>
            </div>
          </div>
        </Card>

        {/* Edit form */}
        <Card className="lg:col-span-2">
          <div className="text-[10px] uppercase tracking-widest text-muted font-mono mb-4">
            {language === "fr" ? "Modifier le profil" : "Edit Profile"}
          </div>
          <div className="space-y-4">
            <div>
              <label className="text-xs text-muted block mb-1">{language === "fr" ? "Nom complet" : "Full name"}</label>
              <input value={fullName} onChange={(e) => setFullName(e.target.value)}
                className="w-full bg-night/40 border border-cardBorder rounded-lg px-3 py-2 text-sm text-sand" />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">{language === "fr" ? "Organisation" : "Organization"}</label>
              <input value={organization} onChange={(e) => setOrganization(e.target.value)}
                className="w-full bg-night/40 border border-cardBorder rounded-lg px-3 py-2 text-sm text-sand"
                placeholder={language === "fr" ? "Optionnel" : "Optional"} />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">{language === "fr" ? "À propos" : "Bio"}</label>
              <textarea value={bio} onChange={(e) => setBio(e.target.value)} rows={3}
                className="w-full bg-night/40 border border-cardBorder rounded-lg px-3 py-2 text-sm text-sand resize-none"
                placeholder={language === "fr" ? "Quelques mots sur ton rôle..." : "A few words about your role..."} />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">{language === "fr" ? "Email" : "Email"}</label>
              <input value={user?.email || ""} disabled
                className="w-full bg-night/20 border border-cardBorder rounded-lg px-3 py-2 text-sm text-muted cursor-not-allowed" />
            </div>

            {error && <div className="text-clay text-xs">{error}</div>}

            <button
              onClick={handleSave}
              disabled={saving}
              className="btn-primary inline-flex items-center gap-2 px-4 py-2 text-sm disabled:opacity-60"
            >
              {saving ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
              {saved
                ? (language === "fr" ? "Enregistré" : "Saved")
                : (language === "fr" ? "Enregistrer" : "Save changes")}
            </button>
          </div>
        </Card>
      </div>

      {/* Activity heatmap */}
      <Card className="mt-5">
        <div className="flex items-center justify-between mb-1">
          <div className="text-[10px] uppercase tracking-widest text-muted font-mono">
            {language === "fr" ? "Activité — 12 derniers mois" : "Activity — Last 12 Months"}
          </div>
          {activity && (
            <div className="text-xs text-muted">
              {activity.total} {language === "fr" ? "actions enregistrées" : "actions logged"}
            </div>
          )}
        </div>
        {activity ? (
          <>
            <ActivityHeatmap counts={activity.counts} language={language} />
            {activity.total < 5 && (
              <p className="text-muted text-xs mt-3 italic">
                {language === "fr"
                  ? "Le suivi d'activité vient de démarrer — ton historique se remplira au fil de ton usage réel de SAHELI, rien n'est préchargé."
                  : "Activity tracking just started — your history will fill in from real SAHELI usage going forward; nothing here is pre-filled."}
              </p>
            )}
          </>
        ) : (
          <div className="text-muted text-sm py-6 text-center">
            {language === "fr" ? "Chargement de l'activité..." : "Loading activity..."}
          </div>
        )}
      </Card>
    </div>
  );
}
