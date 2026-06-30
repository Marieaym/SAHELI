import { createContext, useContext, useState, useEffect } from "react";
import { api } from "../api/client";

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem("saheli_token"));
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem("saheli_user");
    return stored ? JSON.parse(stored) : null;
  });
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (token) {
      api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
    } else {
      delete api.defaults.headers.common["Authorization"];
    }
    setReady(true);
  }, [token]);

  function login(newToken, newUser) {
    localStorage.setItem("saheli_token", newToken);
    localStorage.setItem("saheli_user", JSON.stringify(newUser));
    setToken(newToken);
    setUser(newUser);
  }

  function updateUser(newUser) {
    localStorage.setItem("saheli_user", JSON.stringify(newUser));
    setUser(newUser);
  }

  function logout() {
    localStorage.removeItem("saheli_token");
    localStorage.removeItem("saheli_user");
    setToken(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ token, user, login, logout, updateUser, isAuthenticated: !!token, ready }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
