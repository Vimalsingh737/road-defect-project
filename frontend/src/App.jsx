import { useState, useEffect } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const AI_URL = "http://127.0.0.1:8000";
const BACKEND_URL = "http://localhost:5000/api/reports";

const createPinIcon = (priority, status) => {
  const color =
    status === "Resolved"
      ? "#3b82f6"
      : priority === "Critical"
      ? "#ef4444"
      : priority === "Medium"
      ? "#f97316"
      : "#22c55e";
  return L.divIcon({
    className: "custom-pin",
    html: `<div style="background:${color};width:22px;height:22px;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.5);"></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
};

function LocationPicker({ setCoords }) {
  useMapEvents({
    click(e) {
      setCoords([e.latlng.lat, e.latlng.lng]);
    },
  });
  return null;
}

export default function App() {
  const [mode, setMode] = useState("image");
  const [file, setFile] = useState(null);
  const [conf, setConf] = useState(0.15);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [coords, setCoords] = useState([25.2425, 86.9842]);
  const [reports, setReports] = useState([]);

  useEffect(() => {
    axios.get(BACKEND_URL).then((res) => setReports(res.data)).catch(console.error);
  }, []);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please select a photo or video first!");

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const endpoint = mode === "image" ? "/predict-image" : "/predict-video";
      const aiRes = await axios.post(`${AI_URL}${endpoint}?conf=${conf}`, formData);
      setResult(aiRes.data);

      const savedRes = await axios.post(BACKEND_URL, {
        lat: coords[0],
        lng: coords[1],
        ...aiRes.data,
      });
      setReports((prev) => [savedRes.data, ...prev]);
    } catch (err) {
      alert("Make sure both FastAPI (port 8000) and Node Backend (port 5000) are running!");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const toggleStatus = async (id) => {
    const res = await axios.patch(`${BACKEND_URL}/${id}`);
    setReports((prev) => prev.map((r) => (r.id === id ? res.data : r)));
  };

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", background: "#0f172a", minHeight: "100vh", color: "#f8fafc", padding: "24px" }}>
      <header style={{ maxWidth: "1280px", margin: "0 auto 20px", borderBottom: "1px solid #334155", paddingBottom: "14px" }}>
        <h1 style={{ margin: 0, fontSize: "26px", color: "#38bdf8" }}>
          Road Surface Segmentation & Severity Estimator
        </h1>
        <p style={{ margin: "6px 0 0", color: "#94a3b8", fontSize: "14px" }}>
          YOLOv8-Seg ONNX Engine • FastAPI + Node.js + React Leaflet GIS Dashboard
        </p>
      </header>

      <div style={{ maxWidth: "1280px", margin: "0 auto", display: "grid", gridTemplateColumns: "1fr 1.3fr", gap: "24px" }}>
        <div style={{ background: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155" }}>
          <div style={{ display: "flex", gap: "10px", marginBottom: "16px" }}>
            <button
              type="button"
              onClick={() => { setMode("image"); setFile(null); setResult(null); }}
              style={{ flex: 1, padding: "10px", borderRadius: "8px", border: "none", cursor: "pointer", fontWeight: "600", background: mode === "image" ? "#0284c7" : "#334155", color: "white" }}
            >
              Photo Detection
            </button>
            <button
              type="button"
              onClick={() => { setMode("video"); setFile(null); setResult(null); }}
              style={{ flex: 1, padding: "10px", borderRadius: "8px", border: "none", cursor: "pointer", fontWeight: "600", background: mode === "video" ? "#0284c7" : "#334155", color: "white" }}
            >
              Video Detection
            </button>
          </div>

          <form onSubmit={handleAnalyze}>
            <input
              type="file"
              accept={mode === "image" ? "image/*" : "video/*"}
              onChange={(e) => setFile(e.target.files[0])}
              style={{ width: "100%", marginBottom: "14px", padding: "10px", background: "#0f172a", borderRadius: "8px", border: "1px solid #475569" }}
            />

            <label style={{ display: "block", fontSize: "13px", marginBottom: "6px", color: "#cbd5e1" }}>
              Confidence Threshold: <strong>{conf}</strong>
            </label>
            <input
              type="range"
              min="0.05"
              max="0.70"
              step="0.05"
              value={conf}
              onChange={(e) => setConf(parseFloat(e.target.value))}
              style={{ width: "100%", marginBottom: "14px" }}
            />

            <button
              type="submit"
              disabled={loading}
              style={{ width: "100%", padding: "12px", borderRadius: "8px", border: "none", fontWeight: "700", background: loading ? "#475569" : "#10b981", color: "white", cursor: "pointer" }}
            >
              {loading ? "Running YOLOv8 Segmentation..." : `Analyze ${mode === "image" ? "Photo" : "Video"}`}
            </button>
          </form>

          {result && (
            <div style={{ marginTop: "20px", borderTop: "1px solid #334155", paddingTop: "16px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px", marginBottom: "14px", textAlign: "center" }}>
                <div style={{ background: "#0f172a", padding: "10px", borderRadius: "8px" }}>
                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>Damaged Area</div>
                  <div style={{ fontSize: "18px", fontWeight: "700", color: "#38bdf8" }}>{result.damaged_area_percent}%</div>
                </div>
                <div style={{ background: "#0f172a", padding: "10px", borderRadius: "8px" }}>
                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>Severity</div>
                  <div style={{ fontSize: "18px", fontWeight: "700" }}>{result.severity_score}/100</div>
                </div>
                <div style={{ background: "#0f172a", padding: "10px", borderRadius: "8px" }}>
                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>Priority</div>
                  <div style={{ fontSize: "18px", fontWeight: "700", color: result.priority === "Critical" ? "#ef4444" : result.priority === "Medium" ? "#f97316" : "#22c55e" }}>
                    {result.priority}
                  </div>
                </div>
              </div>

              {result.media_type === "image" ? (
                <img src={result.annotated_image_base64} alt="Segmented" style={{ width: "100%", borderRadius: "8px" }} />
              ) : (
                <video src={result.video_url} controls autoPlay loop style={{ width: "100%", borderRadius: "8px" }} />
              )}
            </div>
          )}
        </div>

        <div style={{ background: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155" }}>
          <h3 style={{ margin: "0 0 10px", fontSize: "17px" }}>
            Live GIS Repair Priority Map (Click map to place pin: {coords[0].toFixed(4)}, {coords[1].toFixed(4)})
          </h3>
          <div style={{ height: "520px", borderRadius: "10px", overflow: "hidden" }}>
            <MapContainer center={coords} zoom={14} style={{ height: "100%", width: "100%" }}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              <LocationPicker setCoords={setCoords} />
              {reports.map((r) => (
                <Marker key={r.id} position={[r.lat, r.lng]} icon={createPinIcon(r.priority, r.status)}>
                  <Popup>
                    <div style={{ width: "200px", color: "#0f172a" }}>
                      <strong>{r.priority} ({r.severity_score}/100)</strong>
                      <p style={{ margin: "4px 0", fontSize: "12px" }}>Area: {r.damaged_area_percent}% | {r.status}</p>
                      {r.annotated_image_base64 && (
                        <img src={r.annotated_image_base64} alt="Pin" style={{ width: "100%", borderRadius: "6px", marginBottom: "6px" }} />
                      )}
                      <button
                        onClick={() => toggleStatus(r.id)}
                        style={{ width: "100%", padding: "6px", background: "#0284c7", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
                      >
                        Mark as {r.status === "Resolved" ? "Pending" : "Resolved"}
                      </button>
                    </div>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          </div>
        </div>
      </div>
    </div>
  );
}