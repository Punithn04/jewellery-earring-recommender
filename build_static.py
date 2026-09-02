"""Generate a fully static demo (no backend) into docs/ for GitHub Pages / any
static host.

Ranking the 5 provided necklaces only touches the precomputed feature index
(outputs/index.pkl) + numpy -- no torch -- so every recommendation can be baked
into a JSON file and served as flat files.

    python precompute.py      # once, builds outputs/index.pkl
    python build_static.py    # writes docs/{index.html,data.json,images/*}
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.index import IMAGES, load, recommend

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs"


def build():
    items = load()
    OUT.mkdir(exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)

    for it in items.values():
        shutil.copyfile(IMAGES / it.image_file, OUT / "images" / f"{it.id}.jpg")

    necklaces = [it for it in items.values() if it.product_type.lower().startswith("neck")]
    data = {"weights": None, "necklaces": []}
    for n in necklaces:
        res = recommend(n.id, items, top_k=15)
        data["weights"] = res["weights"]
        data["necklaces"].append({
            "id": n.id,
            "why": res["query"]["why"],
            "results": res["results"],
        })

    (OUT / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (OUT / "index.html").write_text(HTML, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {OUT}/  ({len(necklaces)} necklaces, {len(items)} images)")


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Earrings for a Necklace</title>
<style>
  :root { font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
  body { margin: 0; background: #faf7f5; color: #2b2b2b; }
  header { padding: 20px 28px; background: #fff; border-bottom: 1px solid #eee; }
  h1 { margin: 0; font-size: 19px; }
  p.sub { margin: 4px 0 0; color: #888; font-size: 13px; }
  main { padding: 24px 28px; max-width: 1100px; margin: 0 auto; }
  .necklaces { display: flex; gap: 12px; flex-wrap: wrap; }
  .necklaces figure { margin: 0; cursor: pointer; border: 3px solid transparent;
    border-radius: 10px; padding: 4px; background: #fff; transition: .15s; }
  .necklaces figure:hover { border-color: #d9c3a5; }
  .necklaces figure.active { border-color: #b5892f; }
  .necklaces img { width: 120px; height: 120px; object-fit: contain; display: block; }
  .necklaces figcaption { text-align: center; font-size: 12px; color: #666; }
  .row { display: flex; align-items: center; gap: 14px; margin: 22px 0 8px; }
  input[type=number] { padding: 6px 8px; font-size: 14px; }
  .chips span { display: inline-block; background: #efe7db; color: #6a5326;
    font-size: 11px; padding: 2px 8px; border-radius: 20px; margin: 2px 4px 2px 0; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px; margin-top: 14px; }
  .card { background: #fff; border: 1px solid #eee; border-radius: 10px; padding: 10px; }
  .card img { width: 100%; height: 170px; object-fit: contain; }
  .card .id { font-weight: 600; font-size: 13px; margin-top: 6px; }
  .bar { height: 6px; background: #eee; border-radius: 4px; overflow: hidden; margin: 6px 0; }
  .bar > i { display: block; height: 100%; background: #b5892f; }
  .muted { color: #999; font-size: 11px; }
  .hint { color: #aaa; font-size: 12px; }
  code { background:#f0ece7; padding:1px 5px; border-radius:4px; font-size:12px; }
  button#go { padding: 8px 18px; font-size: 14px; background: #b5892f; color: #fff;
    border: 0; border-radius: 6px; cursor: pointer; }
  button#go:disabled { opacity: .45; cursor: default; }
  .empty { color: #b7b7b7; font-size: 13px; padding: 18px 0; }
</style>
</head>
<body>
<header>
  <h1>Matching earrings for a selected necklace</h1>
  <p class="sub">Fashion-CLIP embedding (0.55) + CIELAB colour palette (0.30) + zero-shot attribute agreement (0.15) &nbsp;·&nbsp; static demo of the precomputed results</p>
</header>
<main>
  <h3>1 &middot; Pick a necklace</h3>
  <div class="necklaces" id="necklaces"></div>

  <div class="row">
    <label>top <input type="number" id="top" value="5" min="1" max="15" style="width:56px"></label>
    <span class="hint">Live API version (with image upload) is in the repo: <code>uvicorn app:app</code></span>
  </div>

  <div id="queryWhy" class="chips"></div>

  <h3>2 &middot; Recommended earrings</h3>
  <div class="grid" id="results"></div>
</main>

<script>
const $ = s => document.querySelector(s);
let DATA = null, selected = null;

function chips(arr){ return arr.map(c => `<span>${c}</span>`).join(''); }

function render(){
  if(!selected) return;
  const n = DATA.necklaces.find(x => x.id === selected);
  const k = Math.max(1, Math.min(15, +$('#top').value || 5));
  $('#queryWhy').innerHTML = '<b>Query profile:</b> ' + chips(n.why);
  $('#results').innerHTML = n.results.slice(0, k).map(r => `
    <div class="card">
      <img src="images/${r.id}.jpg" alt="${r.id}">
      <div class="id">#${r.rank} &middot; ${r.id} &middot; <span style="color:#b5892f">${r.score.toFixed(3)}</span></div>
      <div class="bar"><i style="width:${(r.score*100).toFixed(0)}%"></i></div>
      <div class="muted">clip ${r.breakdown.clip.toFixed(2)} &middot; colour ${r.breakdown.colour.toFixed(2)} &middot; attr ${r.breakdown.attr.toFixed(2)}</div>
      <div class="chips">${chips(r.why)}</div>
    </div>`).join('');
}

fetch('data.json').then(r => r.json()).then(d => {
  DATA = d;
  const box = $('#necklaces');
  d.necklaces.forEach((p, idx) => {
    const fig = document.createElement('figure');
    fig.innerHTML = `<img src="images/${p.id}.jpg"><figcaption>${p.id}</figcaption>`;
    fig.onclick = () => {
      selected = p.id;
      document.querySelectorAll('.necklaces figure').forEach(f => f.classList.remove('active'));
      fig.classList.add('active');
      render();
    };
    box.appendChild(fig);
    if (idx === 0) fig.click();
  });
});
$('#top').addEventListener('input', render);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build()
