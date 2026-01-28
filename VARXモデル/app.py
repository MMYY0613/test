from __future__ import annotations
import io, sqlite3, pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, Response, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import hashlib

# --- 設定 ---
DB_PATH = Path("./data/app.db")
app = Flask(__name__)

# --- デザイン（デフォルトをライトモードに変更） ---
STYLE = """
<style>
  /* 基本をライトモード(白背景)に設定 */
  :root { 
    --bg: #f5f7fa; --card-bg: #ffffff; --text: #2f3336; 
    --border: #d1d5db; --h3: #1a56db; --nav-bg: #ffffff;
  }
  /* ダークモードクラスが当たった時の色 */
  body.dark-mode { 
    --bg: #0d0e12; --card-bg: #18191c; --text: #b9bbbe; 
    --border: #2f3136; --h3: #5794f2; --nav-bg: #18191c;
  }

  body { background: var(--bg); color: var(--text); font-family: 'Consolas', monospace; margin: 25px; transition: 0.3s; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
  .card { border: 1px solid var(--border); padding: 15px; background: var(--card-bg); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
  .card h3 { margin: 0 0 10px 0; font-size: 13px; color: var(--h3); border-left: 3px solid #f05a28; padding-left: 10px; }
  img { width: 100%; border-radius: 4px; border: 1px solid var(--border); }
  
  .nav { margin-bottom: 20px; padding: 15px; background: var(--nav-bg); border-radius: 8px; border: 1px solid var(--border); display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  input, select { background: var(--bg); color: var(--text); border: 1px solid var(--border); padding: 5px 8px; border-radius: 4px; font-family: inherit; }
  .btn { background: #f05a28; color: white; border: none; padding: 8px 15px; cursor: pointer; border-radius: 4px; font-weight: bold; }
  .y-btn { background: var(--bg); color: var(--text); border: 1px solid var(--border); padding: 4px 10px; cursor: pointer; border-radius: 4px; font-size: 11px; font-weight: bold; }
  
  .mode-switch { margin-left: auto; background: #444; color: #fff; border: none; padding: 6px 15px; border-radius: 20px; cursor: pointer; font-size: 11px; font-weight: bold; }
  body:not(.dark-mode) .mode-switch { background: #ddd; color: #333; }

  .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 10px; overflow-x: auto; }
  .tab-item { padding: 8px 20px; cursor: pointer; border-radius: 4px; text-decoration: none; color: #888; font-size: 12px; font-weight: bold; white-space: nowrap; }
  .tab-item.active { background: var(--border); color: var(--text); border-bottom: 2px solid #f05a28; }
</style>
"""

EXCLUDE_TABLES = ["sqlite_sequence", "all_q_merged_raw", "timeseries_long"]

def get_con():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def get_color_from_name(name: str):
    colors = ["#5794f2", "#47a447", "#f05a28", "#8e44ad", "#d35400", "#2980b9", "#27ae60", "#c0392b"]
    idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(colors)
    return colors[idx]

@app.get("/")
def index():
    with get_con() as con:
        all_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con)['name'].tolist()
    user_tables = [t for t in all_tables if t not in EXCLUDE_TABLES]
    
    target_tab = request.args.get("tab", "")
    dfrom, dto = request.args.get("from", ""), request.args.get("to", "")
    mode = request.args.get("mode", "level")

    years = [2000, 2010, 2015, 2020, 2024, 2025]
    year_btns = "".join([f'<button type="button" class="y-btn" onclick="setYear({y})">{y}年</button>' for y in years])

    tab_links = [f'<a href="/?from={dfrom}&to={dto}&mode={mode}" class="tab-item {"active" if not target_tab else ""}">ALL VIEW</a>']
    for t in user_tables:
        tab_links.append(f'<a href="/?tab={t}&from={dfrom}&to={dto}&mode={mode}" class="tab-item {"active" if target_tab == t else ""}">{t.upper()}</a>')

    display_tables = [target_tab] if target_tab else user_tables
    all_cards = []
    with get_con() as con:
        for table in display_tables:
            series_names = pd.read_sql(f"SELECT DISTINCT series FROM {table}", con)['series'].tolist()
            for s in series_names:
                # <img>タグのURLに現在選択中のテーマ（darkかどうか）をJSで渡す仕組みを追加
                all_cards.append(f'''
                    <div class="card">
                        <h3><span style="color:#888; font-size:11px; margin-right:5px;">[{table.upper()}]</span>{s}</h3>
                        <img class="plot-img" data-base="/plot?table={table}&series={s}&from={dfrom}&to={dto}&mode={mode}" src="">
                    </div>''')

    return f"""
    <html><head><title>Analyst Dashboard</title>{STYLE}
    <script>
      function setYear(y) {{
        document.getElementById('f').value = y + '-01-01';
        document.getElementById('t').value = y + '-12-31';
        document.getElementById('search-form').submit();
      }}
      function updateGraphs() {{
        const isDark = document.body.classList.contains('dark-mode');
        document.querySelectorAll('.plot-img').forEach(img => {{
            img.src = img.getAttribute('data-base') + '&theme=' + (isDark ? 'dark' : 'light');
        }});
      }}
      function toggleMode() {{
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('theme', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
        updateGraphs();
      }}
      window.onload = () => {{
        if(localStorage.getItem('theme') === 'dark') document.body.classList.add('dark-mode');
        updateGraphs();
      }};
    </script>
    </head><body>
        <div class="nav">
            <form id="search-form" style="display:flex; gap:10px; align-items:center; margin:0;">
                <input type="hidden" name="tab" value="{target_tab}">
                FROM: <input type="date" name="from" id="f" value="{dfrom}"> 
                TO: <input type="date" name="to" id="t" value="{dto}">
                <select name="mode" onchange="this.form.submit()">
                    <option value="level" {"selected" if mode=="level" else ""}>Level</option>
                    <option value="diff" {"selected" if mode=="diff" else ""}>Diff</option>
                    <option value="pct_chg" {"selected" if mode=="pct_chg" else ""}>Pct Change</option>
                    <option value="log_diff" {"selected" if mode=="log_diff" else ""}>Log Diff</option>
                </select>
                <button type="submit" class="btn">APPLY</button>
                <a href="/?tab={target_tab}&mode={mode}" style="color:#888; font-size:12px; text-decoration:none;">Reset Date</a>
                <div style="margin-left:10px; padding-left:15px; border-left:1px solid var(--border); display:flex; gap:5px;">
                    {year_btns}
                </div>
            </form>
            <button class="mode-switch" onclick="toggleMode()">🌓 Switch Theme</button>
        </div>
        <div class="tabs">{ " ".join(tab_links) }</div>
        <div class="grid">{ "".join(all_cards) }</div>
    </body></html>
    """

@app.get("/plot")
def plot():
    table, series = request.args.get("table"), request.args.get("series")
    dfrom, dto = request.args.get("from"), request.args.get("to")
    mode = request.args.get("mode", "level")
    theme = request.args.get("theme", "light") # テーマ判定
    
    # テーマに応じた色設定
    bg_color = "#18191c" if theme == "dark" else "#ffffff"
    text_color = "#cccccc" if theme == "dark" else "#333333"
    grid_color = "#444" if theme == "dark" else "#ddd"

    params, where = [series], "series = ?"
    if dfrom: where += " AND date >= ?"; params.append(dfrom)
    if dto: where += " AND date <= ?"; params.append(dto)
        
    with get_con() as con:
        df = pd.read_sql_query(f'SELECT date as "Date", value as y FROM {table} WHERE {where} ORDER BY date ASC', con, params=params)

    if df.empty: return "No Data", 404
    df["Date"] = pd.to_datetime(df["Date"])
    
    if mode == "diff": df["y"] = df["y"].diff()
    elif mode == "pct_chg": df["y"] = df["y"].pct_change() * 100
    elif mode == "log_diff":
        if (df["y"] <= 0).any():
            return _draw_error("Log Diff Error\n(Need > 0)", bg_color)
        df["y"] = np.log(df["y"]).diff()

    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    line_color = get_color_from_name(series)

    ax.plot(df["Date"], df["y"], color=line_color, lw=2.2, antialiased=True)
    ax.axhline(0, color='#888', lw=1.0, alpha=0.5)

    ax.tick_params(axis='both', colors=text_color, labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontweight('bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.grid(True, color=grid_color, ls="--", lw=0.6, alpha=0.6)
    for s in ax.spines.values(): s.set_edgecolor(grid_color)
    
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, facecolor=bg_color)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")

def _draw_error(msg, bg_color):
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.text(0.5, 0.5, msg, color="#ff4d4f", ha="center", va="center", transform=ax.transAxes, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, facecolor=bg_color)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)