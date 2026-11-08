const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json({ limit: "25mb" }));

const DB_FILE = path.join(__dirname, "reports.json");

const readReports = () => {
  if (!fs.existsSync(DB_FILE)) return [];
  return JSON.parse(fs.readFileSync(DB_FILE, "utf-8"));
};

const writeReports = (data) => {
  fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2));
};

// Health check route for Render cloud deployment
app.get("/", (req, res) => {
  res.json({ status: "online", service: "Road Defect Node.js Backend" });
});

// Get all saved road defect reports
app.get("/api/reports", (req, res) => {
  res.json(readReports());
});

// Save a new road defect report from AI inference
app.post("/api/reports", (req, res) => {
  const reports = readReports();
  const newReport = {
    id: Date.now(),
    status: "Pending Repair",
    createdAt: new Date().toLocaleString(),
    ...req.body,
  };
  reports.unshift(newReport);
  writeReports(reports);
  res.status(201).json(newReport);
});

// Toggle repair status (Pending Repair <-> Resolved)
app.patch("/api/reports/:id", (req, res) => {
  const reports = readReports();
  const idx = reports.findIndex((r) => r.id === Number(req.params.id));
  if (idx === -1) return res.status(404).json({ error: "Not found" });

  reports[idx].status =
    reports[idx].status === "Resolved" ? "Pending Repair" : "Resolved";
  writeReports(reports);
  res.json(reports[idx]);
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Node.js Backend running on port ${PORT}`);
});