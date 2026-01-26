from __future__ import annotations
import io, sqlite3, pandas as pd
from pathlib import Path
from flask import Flask, Response, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# --- 設定 ---
DB_PATH = Path("./data/app.db")
TABLE = "all_q_merged_raw"
app = Flask(__name__)

# --- スタイル（今のデザインを完全維持） ---
STYLE = """
<style>
  body { background: #0d0e12; color: #b9bbbe; font-family: 'Consolas', monospace; margin: 25px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; }
  .card { border: 1px solid #2f3136; padding: 15px; background: #18191c; border-radius: 8px; }
  .card h3 { margin: 0 0 10px 0; font-size: 14px; color: #5794f2; border-left: 3px solid #f05a28; padding-left: 10px; }
  img { width: 100%; border-radius: 4px; }
  .nav { margin-bottom: 30px; padding: 15px; background: #18191c; border-radius: 8px; border: 1px solid #2f3136; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
  input { background: #202225; color: #fff; border: 1px solid #444; padding: 5px 8px; border-radius: 4px; }
  .btn { background: #f05a28; color: white; border: none; padding: 8px 20px; cursor: pointer; border-radius: 4px; font-weight: bold; }
  .btn:hover { background: #ff784e; }
  /* 年次ボタン用：今のデザインに馴染むように調整 */
  .y-btn { background: #2f3136; color: #eee; border: 1px solid #444; padding: 4px 10px; cursor: pointer; border-radius: 4px; font-size: 12px; font-weight: bold; }
  .y-btn:hover { border-color: #f05a28; background: #383a40; }
</style>
"""

def get_con():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@app.get("/")
def index():
    dfrom = request.args.get("from", "")
    dto = request.args.get("to", "")
    
    # 戻りたい年のリスト（よく使うものを並べておくと便利です）
    years = [2000, 2010, 2020, 2024, 2025]
    year_btns = "".join([
        f'<button type="button" class="y-btn" onclick="setYear({y})">{y}年</button>' 
        for y in years
    ])
    
    cols = ["GDP_LOGDIFF", "NIKKEI_LOGRET", "TOPIX_LOGRET", "FX_LOGRET"]
    cards = "".join([f'<div class="card"><h3>{c}</h3><img src="/plot?col={c}&from={dfrom}&to={dto}"></div>' for c in cols])
    
    return f"""
    <html>
      <head>
        <title>RHEL Analyst Dashboard</title>
        {STYLE}
        <script>
          // ボタンを押した時にカレンダーの値を書き換えて自動送信
          function setYear(y) {{
            document.getElementById('f').value = y + '-01-01';
            document.getElementById('t').value = y + '-12-31';
            document.getElementById('search-form').submit();
          }}
        </script>
      </head>
      <body>
        <div class="nav">
          <b style="color:#fff;">📊 DATA VIEW</b>
          <form id="search-form" style="display:flex; align-items:center; gap:10px; margin:0;">
            <span>FROM:</span><input type="date" name="from" id="f" value="{dfrom}">
            <span>TO:</span><input type="date" name="to" id="t" value="{dto}">
            <button type="submit" class="btn">APPLY</button>
            <div style="display:flex; gap:5px; margin-left:10px; padding-left:15px; border-left: 1px solid #333;">
              {year_btns}
            </div>
            <a href="/" style="color:#888; font-size:12px; margin-left:15px; text-decoration:none;">Reset</a>
          </form>
          <div style="font-size:11px; color:#555; margin-left:auto;">DB: {DB_PATH}</div>
        </div>
        <div class="grid">{cards}</div>
      </body>
    </html>
    """

@app.get("/plot")
def plot():
    col = request.args.get("col", "GDP_LOGDIFF")
    dfrom = request.args.get("from")
    dto = request.args.get("to")
    
    where_clauses = [f'"{col}" IS NOT NULL']
    params = []
    
    if dfrom:
        where_clauses.append('"Date" >= ?')
        params.append(dfrom)
    if dto:
        where_clauses.append('"Date" <= ?')
        params.append(dto)
        
    where_sql = " AND ".join(where_clauses)
    
    with get_con() as con:
        df = pd.read_sql_query(
            f'SELECT "Date", "{col}" as y FROM "{TABLE}" WHERE {where_sql} ORDER BY "Date" ASC',
            con, params=params
        )

    if df.empty: return "No Data in this range", 404
    df["Date"] = pd.to_datetime(df["Date"])
    
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#18191c")
    ax.set_facecolor("#18191c")
    
    colors = {"GDP": "#5794f2", "NIKKEI": "#f2cc57", "TOPIX": "#ff7875", "FX": "#95de64"}
    main_color = next((v for k, v in colors.items() if k in col), "#5794f2")

    ax.plot(df["Date"], df["y"], color=main_color, lw=1.8, antialiased=True)
    ax.axhline(0, color='#666', lw=1, ls='-', zorder=1)

    # 軸の白・太字・2015/01形式
    ax.tick_params(axis='both', colors='#eee', labelsize=11) 
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    fig.autofmt_xdate(rotation=0, ha='center')
    ax.grid(True, color="#444", ls="--", lw=0.5, alpha=0.7)
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)