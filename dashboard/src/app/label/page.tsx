"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  FormControl,
  IconButton,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Slider,
  Stack,
  Tooltip,
  Typography,
} from "@mui/material";
import SaveIcon from "@mui/icons-material/Save";
import RefreshIcon from "@mui/icons-material/Refresh";
import ReplayIcon from "@mui/icons-material/Replay";
import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import FolderZipOutlinedIcon from "@mui/icons-material/FolderZipOutlined";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const UNASSIGNED = -1;

type FolderInfo = { name: string; n_images: number; n_labels: number };
type ImageInfo = {
  name: string;
  url: string;
  labelled: boolean;
  n_boxes: number;
  in_images_dir: boolean;
};
type Box4 = [number, number, number, number];
type Box = { id: number; bbox_xyxy: Box4; confidence: number | null; class_id: number | null };
type BoxesResp = {
  image_width: number;
  image_height: number;
  boxes: Box[];
  source: "detect" | "load";
  model?: string;
};
type ModelInfo = { name: string; available: boolean; size_mb: number | null; loaded: boolean };

const BOX_COLORS = ["#5b8def", "#f06292", "#ffb74d", "#81c784", "#ba68c8", "#4dd0e1", "#ff8a65", "#a1887f"];

export default function LabelPage() {
  const [classes, setClasses] = useState<string[]>([]);
  const [folders, setFolders] = useState<FolderInfo[]>([]);
  const [folder, setFolder] = useState<string>("");
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [resp, setResp] = useState<BoxesResp | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [assignments, setAssignments] = useState<Record<number, number>>({});
  const [hovered, setHovered] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [conf, setConf] = useState(0.4);
  const [preparing, setPreparing] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string>("yolo26n");

  const fetchFolders = async () => {
    setError(null);
    try {
      const [foldersRes, classesRes, modelsRes] = await Promise.all([
        fetch(`${API_URL}/label/folders`),
        fetch(`${API_URL}/label/classes`),
        fetch(`${API_URL}/label/models`),
      ]);
      if (!foldersRes.ok) throw new Error(`folders ${foldersRes.status}`);
      const f = (await foldersRes.json()) as { folders: FolderInfo[] };
      const c = (await classesRes.json()) as { classes: string[] };
      const m = (await modelsRes.json()) as { models: ModelInfo[]; default: string };
      setFolders(f.folders);
      setClasses(c.classes);
      setModels(m.models);
      // Only override the user's pick if they haven't changed it from the initial default.
      setModel((prev) => (prev === "yolo26n" ? m.default : prev));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const reloadAll = async () => {
    await fetchFolders();
    if (folder) await fetchImages(folder);
  };

  const fetchImages = async (f: string) => {
    setError(null);
    try {
      const res = await fetch(`${API_URL}/label/images?folder=${encodeURIComponent(f)}`);
      if (!res.ok) throw new Error(`images ${res.status}`);
      const data = (await res.json()) as { images: ImageInfo[] };
      setImages(data.images);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const applyResponse = (data: BoxesResp) => {
    setResp(data);
    setBoxes(data.boxes);
    const init: Record<number, number> = {};
    data.boxes.forEach((b) => (init[b.id] = b.class_id ?? UNASSIGNED));
    setAssignments(init);
  };

  const runLoad = async (f: string, name: string) => {
    setLoading(true);
    setError(null);
    setInfo(null);
    setResp(null);
    setBoxes([]);
    try {
      const res = await fetch(
        `${API_URL}/label/load?folder=${encodeURIComponent(f)}&name=${encodeURIComponent(name)}`,
      );
      if (!res.ok) throw new Error(`load ${res.status}: ${await res.text()}`);
      applyResponse((await res.json()) as BoxesResp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const runDetect = async (f: string, name: string, c: number, m: string) => {
    setLoading(true);
    setError(null);
    setInfo(null);
    setResp(null);
    setBoxes([]);
    try {
      const form = new FormData();
      form.append("folder", f);
      form.append("name", name);
      form.append("conf", String(c));
      form.append("model", m);
      const res = await fetch(`${API_URL}/label/detect`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`detect ${res.status}: ${await res.text()}`);
      applyResponse((await res.json()) as BoxesResp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFolders();
  }, []);

  useEffect(() => {
    if (folder) fetchImages(folder);
  }, [folder]);

  const handleSelectImage = (img: ImageInfo) => {
    setSelected(img.name);
    if (!folder) return;
    if (img.labelled) {
      runLoad(folder, img.name);
    } else {
      runDetect(folder, img.name, conf, model);
    }
  };

  const handleReDetect = () => {
    if (folder && selected) runDetect(folder, selected, conf, model);
  };

  const handleDeleteBox = (id: number) => {
    setBoxes((prev) => prev.filter((b) => b.id !== id));
    setAssignments((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    if (hovered === id) setHovered(null);
  };

  const handleSave = async () => {
    if (!folder || !selected || !resp) return;
    const toSave = boxes
      .filter((b) => assignments[b.id] !== undefined && assignments[b.id] !== UNASSIGNED)
      .map((b) => ({ class_id: assignments[b.id], bbox_xyxy: b.bbox_xyxy }));
    setSaving(true);
    setError(null);
    setInfo(null);
    try {
      const res = await fetch(`${API_URL}/label/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder,
          name: selected,
          boxes: toSave,
          image_width: resp.image_width,
          image_height: resp.image_height,
        }),
      });
      if (!res.ok) throw new Error(`save ${res.status}: ${await res.text()}`);
      const data = (await res.json()) as { saved: string; n_boxes: number };
      setInfo(`Saved ${data.n_boxes} box(es) to ${data.saved}`);
      await reloadAll();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const allAssigned = useMemo(() => {
    if (boxes.length === 0) return false;
    return boxes.every((b) => assignments[b.id] !== UNASSIGNED);
  }, [boxes, assignments]);

  const handlePrepare = async () => {
    if (!folder) return;
    setPreparing(true);
    setError(null);
    setInfo(null);
    try {
      const form = new FormData();
      form.append("folder", folder);
      const res = await fetch(`${API_URL}/label/prepare`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`prepare ${res.status}: ${await res.text()}`);
      const data = (await res.json()) as {
        moved: number;
        already_organised: number;
        missing_image: number;
      };
      setInfo(
        `Prepared: moved ${data.moved} image(s) into images/, ${data.already_organised} already organised, ${data.missing_image} missing.`,
      );
      await reloadAll();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPreparing(false);
    }
  };

  const labelledCount = images.filter((i) => i.labelled).length;
  const selectedImage = images.find((i) => i.name === selected);
  const imageUrl = selectedImage ? `${API_URL}${selectedImage.url}` : null;

  return (
    <Container maxWidth={false} sx={{ py: 3 }}>
      <Stack direction={{ xs: "column", md: "row" }} spacing={2} sx={{ alignItems: "stretch" }}>
        {/* Left panel: folder + image list */}
        <Card sx={{ width: { xs: "100%", md: 280 }, flexShrink: 0 }}>
          <CardContent>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center", mb: 2 }}>
              <FormControl size="small" sx={{ flex: 1 }}>
                <InputLabel>Folder</InputLabel>
                <Select
                  label="Folder"
                  value={folder}
                  onChange={(e) => {
                    setFolder(e.target.value);
                    setSelected(null);
                    setResp(null);
                    setBoxes([]);
                  }}
                >
                  {folders.length === 0 && <MenuItem disabled>No folders in raw_frames/</MenuItem>}
                  {folders.map((f) => (
                    <MenuItem key={f.name} value={f.name}>
                      {f.name} ({f.n_labels}/{f.n_images})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Button onClick={reloadAll} size="small" startIcon={<RefreshIcon />}>
                Reload
              </Button>
            </Stack>

            {folder && (
              <>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {labelledCount} / {images.length} labelled
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={images.length === 0 ? 0 : (labelledCount / images.length) * 100}
                  sx={{ mb: 2 }}
                />
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  startIcon={<FolderZipOutlinedIcon />}
                  onClick={handlePrepare}
                  disabled={preparing || labelledCount === 0}
                  sx={{ mb: 2 }}
                >
                  {preparing ? "Preparing..." : "Prepare for training"}
                </Button>
                <Stack spacing={0.5} sx={{ maxHeight: "60vh", overflowY: "auto" }}>
                  {images.map((img) => (
                    <Paper
                      key={img.name}
                      onClick={() => handleSelectImage(img)}
                      sx={{
                        p: 1,
                        cursor: "pointer",
                        bgcolor:
                          selected === img.name
                            ? "primary.dark"
                            : img.labelled
                              ? "success.dark"
                              : "background.paper",
                        "&:hover": { bgcolor: "action.hover" },
                      }}
                    >
                      <Typography variant="caption" sx={{ display: "block" }} noWrap>
                        {img.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {img.labelled ? `${img.n_boxes} box(es) saved` : "unlabelled"}
                      </Typography>
                    </Paper>
                  ))}
                </Stack>
              </>
            )}
          </CardContent>
        </Card>

        {/* Centre panel: image preview with overlay boxes */}
        <Card sx={{ flex: 1, minWidth: 0 }}>
          <CardContent>
            {!selected && (
              <Typography color="text.secondary">Select an image from the list.</Typography>
            )}
            {selected && (
              <>
                <Stack
                  direction={{ xs: "column", sm: "row" }}
                  spacing={2}
                  sx={{ alignItems: { sm: "center" }, mb: 2 }}
                >
                  <Typography variant="subtitle2" sx={{ flex: 1 }} noWrap>
                    {selected}
                  </Typography>
                  <FormControl size="small" sx={{ minWidth: 140 }}>
                    <InputLabel>Model</InputLabel>
                    <Select
                      label="Model"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                    >
                      {models.map((m) => (
                        <MenuItem key={m.name} value={m.name} disabled={!m.available}>
                          {m.name}
                          {m.size_mb !== null && ` (${m.size_mb} MB)`}
                          {!m.available && " — missing"}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <Box sx={{ width: 220 }}>
                    <Typography variant="caption" color="text.secondary">
                      Detection threshold: {conf.toFixed(2)}
                    </Typography>
                    <Slider
                      size="small"
                      value={conf}
                      min={0.05}
                      max={0.95}
                      step={0.05}
                      onChange={(_, v) => setConf(v as number)}
                    />
                  </Box>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<ReplayIcon />}
                    onClick={handleReDetect}
                    disabled={loading}
                  >
                    Re-detect
                  </Button>
                </Stack>

                {loading && <LinearProgress sx={{ mb: 1 }} />}

                {resp?.source === "load" && (
                  <Alert severity="info" sx={{ mb: 1 }}>
                    Loaded {boxes.length} saved box(es). Re-detect to discard and run fresh detection.
                  </Alert>
                )}

                {imageUrl && resp && (
                  <Box
                    sx={{
                      position: "relative",
                      display: "inline-block",
                      maxWidth: "100%",
                      lineHeight: 0,
                    }}
                  >
                    <Box
                      component="img"
                      src={imageUrl}
                      alt={selected}
                      sx={{ maxWidth: "100%", display: "block", borderRadius: 1 }}
                    />
                    {boxes.map((b, idx) => {
                      const [x1, y1, x2, y2] = b.bbox_xyxy;
                      const left = (x1 / resp.image_width) * 100;
                      const top = (y1 / resp.image_height) * 100;
                      const width = ((x2 - x1) / resp.image_width) * 100;
                      const height = ((y2 - y1) / resp.image_height) * 100;
                      const color = BOX_COLORS[idx % BOX_COLORS.length];
                      const isHovered = hovered === b.id;
                      return (
                        <Box
                          key={b.id}
                          onMouseEnter={() => setHovered(b.id)}
                          onMouseLeave={() => setHovered(null)}
                          sx={{
                            position: "absolute",
                            left: `${left}%`,
                            top: `${top}%`,
                            width: `${width}%`,
                            height: `${height}%`,
                            border: `${isHovered ? 4 : 2}px solid ${color}`,
                            boxShadow: isHovered ? `0 0 0 2px rgba(255,255,255,0.6)` : undefined,
                            transition: "border-width 80ms",
                          }}
                        >
                          <Box
                            sx={{
                              position: "absolute",
                              top: -2,
                              left: -2,
                              bgcolor: color,
                              color: "#000",
                              fontWeight: "bold",
                              px: 0.75,
                              fontSize: 12,
                              lineHeight: 1.4,
                              borderTopLeftRadius: 4,
                            }}
                          >
                            {b.id}
                          </Box>
                        </Box>
                      );
                    })}
                  </Box>
                )}

                {resp && boxes.length === 0 && !loading && (
                  <Alert severity="warning" sx={{ mt: 2 }}>
                    No boxes. Lower the threshold and Re-detect, or move on to the next image.
                  </Alert>
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Right panel: detections + class assignments + save */}
        <Card sx={{ width: { xs: "100%", md: 320 }, flexShrink: 0 }}>
          <CardContent>
            <Typography variant="subtitle2" gutterBottom>
              Detections
            </Typography>

            {!resp && <Typography color="text.secondary">No image selected.</Typography>}

            {resp && (
              <Stack spacing={1} sx={{ mb: 2 }}>
                {boxes.map((b, idx) => {
                  const color = BOX_COLORS[idx % BOX_COLORS.length];
                  return (
                    <Paper
                      key={b.id}
                      onMouseEnter={() => setHovered(b.id)}
                      onMouseLeave={() => setHovered(null)}
                      sx={{
                        p: 1,
                        borderLeft: `4px solid ${color}`,
                        bgcolor: hovered === b.id ? "action.hover" : "background.paper",
                      }}
                    >
                      <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                        <Chip
                          label={`#${b.id}`}
                          size="small"
                          sx={{ bgcolor: color, color: "#000", fontWeight: "bold" }}
                        />
                        {b.confidence !== null && (
                          <Typography variant="caption" color="text.secondary">
                            {(b.confidence * 100).toFixed(0)}%
                          </Typography>
                        )}
                        <Box sx={{ flex: 1 }} />
                        <Tooltip title="Delete this box (e.g. false positive)">
                          <IconButton
                            size="small"
                            onClick={() => handleDeleteBox(b.id)}
                            color="error"
                          >
                            <DeleteOutlinedIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Stack>
                      <FormControl size="small" fullWidth sx={{ mt: 1 }}>
                        <InputLabel>Class</InputLabel>
                        <Select
                          label="Class"
                          value={assignments[b.id] ?? UNASSIGNED}
                          onChange={(e) =>
                            setAssignments((prev) => ({ ...prev, [b.id]: Number(e.target.value) }))
                          }
                        >
                          <MenuItem value={UNASSIGNED}>— unset —</MenuItem>
                          {classes.map((c, i) => (
                            <MenuItem key={c} value={i}>
                              {c}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Paper>
                  );
                })}
              </Stack>
            )}

            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              fullWidth
              disabled={!resp || boxes.length === 0 || !allAssigned || saving}
              onClick={handleSave}
            >
              {saving ? "Saving..." : "Save labels"}
            </Button>
            {!allAssigned && boxes.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
                Assign every detection a class to enable Save.
              </Typography>
            )}

            {info && (
              <Alert severity="success" sx={{ mt: 2 }}>
                {info}
              </Alert>
            )}
            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </CardContent>
        </Card>
      </Stack>

      {loading && !resp && (
        <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
          <CircularProgress />
        </Box>
      )}
    </Container>
  );
}
