export async function getJSON(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).error || (await r.json()).detail || m; } catch (e) { /* keep status text */ }
    throw new Error(m);
  }
  return r.json();
}
export const fileUrl = (id, name) => `/api/scans/${id}/files/${name}`;
export const listScans = () => getJSON("/api/scans");
export const getSummary = (id) => getJSON(`/api/scans/${id}/summary`);
export const getReport = (id) => getJSON(`/api/scans/${id}/report`, { method: "POST" });
export const getHealth = () => getJSON("/api/health");
