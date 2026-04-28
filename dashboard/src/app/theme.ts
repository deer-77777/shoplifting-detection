"use client";

import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#5b8def" },
    secondary: { main: "#f06292" },
    background: { default: "#0e1116", paper: "#161b22" },
  },
  shape: { borderRadius: 10 },
});

export default theme;
