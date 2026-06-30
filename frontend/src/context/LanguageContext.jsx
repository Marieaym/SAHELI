import { createContext, useContext, useState } from "react";
import { translations } from "../i18n/translations";

const LanguageContext = createContext();

export function LanguageProvider({ children }) {
  const [language, setLanguage] = useState(() => localStorage.getItem("saheli_lang") || "en");

  function changeLanguage(lang) {
    localStorage.setItem("saheli_lang", lang);
    setLanguage(lang);
  }

  function t(path, vars) {
    const keys = path.split(".");
    let value = translations[language];
    for (const k of keys) {
      value = value?.[k];
    }
    if (value === undefined) {
      // Fallback to English, then to the key itself, so a missing translation
      // never renders as a blank label.
      let fallback = translations.en;
      for (const k of keys) fallback = fallback?.[k];
      value = fallback ?? path;
    }
    if (vars && typeof value === "string") {
      return value.replace(/\{(\w+)\}/g, (_, key) => vars[key] ?? `{${key}}`);
    }
    return value;
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage: changeLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}
