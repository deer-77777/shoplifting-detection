"use client";

import { AppBar, Tab, Tabs, Toolbar, Typography } from "@mui/material";
import VisibilityIcon from "@mui/icons-material/Visibility";
import { usePathname, useRouter } from "next/navigation";

export default function Nav() {
  const router = useRouter();
  const pathname = usePathname();
  const value = pathname?.startsWith("/label") ? "/label" : "/";

  return (
    <AppBar position="static" elevation={0}>
      <Toolbar sx={{ gap: 3 }}>
        <VisibilityIcon />
        <Typography variant="h6" sx={{ flexShrink: 0 }}>
          Shoplifting Detection
        </Typography>
        <Tabs
          value={value}
          onChange={(_, v) => router.push(v)}
          textColor="inherit"
          indicatorColor="secondary"
        >
          <Tab label="Predict" value="/" />
          <Tab label="Labelling" value="/label" />
        </Tabs>
      </Toolbar>
    </AppBar>
  );
}
