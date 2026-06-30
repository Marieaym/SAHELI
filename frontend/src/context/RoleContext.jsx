import { createContext, useContext, useState } from "react";

const RoleContext = createContext();

export const ROLES = { farmer: { icon: "🌾" }, ngo: { icon: "📡" }, minister: { icon: "🏛️" } };

export const ROLE_NAV = {
  farmer: ["/", "/map", "/alerts", "/assistant"],
  ngo: null,
  minister: ["/", "/pipeline", "/map", "/feed", "/brief", "/intervention", "/assistant", "/validation", "/causal"],
};

export function RoleProvider({ children }) {
  const [role, setRoleState] = useState(() => localStorage.getItem("saheli_role") || "ngo");
  const [homeDistrict, setHomeDistrictState] = useState(() => localStorage.getItem("saheli_home_district") || "");

  const setRole = (r) => { localStorage.setItem("saheli_role", r); setRoleState(r); };
  const setHomeDistrict = (d) => { localStorage.setItem("saheli_home_district", d); setHomeDistrictState(d); };

  return (
    <RoleContext.Provider value={{ role, setRole, homeDistrict, setHomeDistrict }}>
      {children}
    </RoleContext.Provider>
  );
}

export function useRole() {
  return useContext(RoleContext);
}
