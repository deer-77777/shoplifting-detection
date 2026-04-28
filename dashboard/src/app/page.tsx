"use client";

import { useRef, useState } from "react";
import {
  AppBar,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  LinearProgress,
  Paper,
  Slider,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Toolbar,
  Typography,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import VisibilityIcon from "@mui/icons-material/Visibility";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
};

type PredictResponse = {
  detections: Detection[];
  count: number;
  annotated_image_b64: string;
  image_width: number;
  image_height: number;
};

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [conf, setConf] = useState(0.25);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePickFile = () => fileInputRef.current?.click();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/predict?conf=${conf}`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        throw new Error(`API ${res.status}: ${await res.text()}`);
      }
      setResult((await res.json()) as PredictResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const annotatedSrc = result
    ? `data:image/png;base64,${result.annotated_image_b64}`
    : null;

  return (
    <>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <VisibilityIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Shoplifting Detection Dashboard</Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Stack spacing={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                1. Upload an image
              </Typography>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ alignItems: "center" }}>
                <Button
                  variant="contained"
                  startIcon={<CloudUploadIcon />}
                  onClick={handlePickFile}
                >
                  Choose image
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={handleFileChange}
                />
                <Typography variant="body2" color="text.secondary">
                  {file ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)` : "No file selected"}
                </Typography>
              </Stack>

              <Box sx={{ mt: 3, maxWidth: 360 }}>
                <Typography variant="body2" gutterBottom>
                  Confidence threshold: {conf.toFixed(2)}
                </Typography>
                <Slider
                  value={conf}
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  onChange={(_, v) => setConf(v as number)}
                />
              </Box>

              <Button
                variant="contained"
                color="secondary"
                disabled={!file || loading}
                onClick={handlePredict}
                sx={{ mt: 2 }}
              >
                {loading ? "Detecting..." : "Run detection"}
              </Button>
              {loading && <LinearProgress sx={{ mt: 2 }} />}
              {error && (
                <Paper sx={{ mt: 2, p: 2, bgcolor: "error.dark" }}>
                  <Typography variant="body2">{error}</Typography>
                </Paper>
              )}
            </CardContent>
          </Card>

          {(previewUrl || annotatedSrc) && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  2. Result
                </Typography>
                <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                  {previewUrl && (
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="overline">Original</Typography>
                      <Box
                        component="img"
                        src={previewUrl}
                        alt="upload preview"
                        sx={{ width: "100%", borderRadius: 2, border: 1, borderColor: "divider" }}
                      />
                    </Box>
                  )}
                  {annotatedSrc && (
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="overline">Annotated</Typography>
                      <Box
                        component="img"
                        src={annotatedSrc}
                        alt="annotated"
                        sx={{ width: "100%", borderRadius: 2, border: 1, borderColor: "divider" }}
                      />
                    </Box>
                  )}
                </Stack>
              </CardContent>
            </Card>
          )}

          {result && (
            <Card>
              <CardContent>
                <Stack direction="row" spacing={2} sx={{ alignItems: "center", mb: 2 }}>
                  <Typography variant="h6">3. Detections</Typography>
                  <Chip
                    label={`${result.count} found`}
                    color={result.count > 0 ? "secondary" : "default"}
                  />
                  {result.detections.some((d) => d.class_name === "Shoplifting") && (
                    <Chip label="ALERT: Shoplifting suspected" color="error" />
                  )}
                </Stack>

                {result.count === 0 ? (
                  <Typography color="text.secondary">
                    No detections above the confidence threshold.
                  </Typography>
                ) : (
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>#</TableCell>
                        <TableCell>Class</TableCell>
                        <TableCell align="right">Confidence</TableCell>
                        <TableCell>Bounding box (x1, y1, x2, y2)</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {result.detections.map((d, i) => (
                        <TableRow key={i}>
                          <TableCell>{i + 1}</TableCell>
                          <TableCell>
                            <Chip
                              size="small"
                              label={d.class_name}
                              color={d.class_name === "Shoplifting" ? "error" : "primary"}
                            />
                          </TableCell>
                          <TableCell align="right">{(d.confidence * 100).toFixed(1)}%</TableCell>
                          <TableCell>
                            {d.bbox_xyxy.map((v) => v.toFixed(0)).join(", ")}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          )}
        </Stack>
      </Container>
    </>
  );
}
