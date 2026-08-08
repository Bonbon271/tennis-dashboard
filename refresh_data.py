"""
refresh_data.py
================
Pobiera swieze dane meczowe ATP (2023-2026) oraz trwajace turnieje
(ongoing_tourneys.csv) i podmienia 4 bloki danych osadzone w pliku dashboardu
(ACE_DATA, H2H_DATA, EXTRA_STATS, MATCH_ACES_STATS), nie ruszajac reszty
pliku (CSS, JS, uklad strony).

Uzycie:
    python3 refresh_data.py

Wymaga pliku szablonu 'index.html' w tym samym katalogu
(czyli obecnej wersji dashboardu) - skrypt go nadpisuje swiezymi danymi.

Zrodlo danych: stats.tennismylife.org (oficjalna, na biezaco aktualizowana
strona projektu TennisMyLife - NIE ich repo na GitHubie, ktore autorzy sami
oznaczyli jako archiwalne/nieaktualizowane, przez co pomijalo m.in. caly
Wimbledon 2026). Format kolumn identyczny jak w projekcie Jeffa Sackmanna.
UWAGA LICENCYJNA: strona deklaruje licencje MIT, ale wczesniejsze repo
GitHub tego samego projektu deklarowalo CC BY-NC-SA (non-commercial) -
przed uzyciem komercyjnym warto to wyjasnic bezposrednio z autorami
(kontakt: infotennismylife@gmail.com).
"""

import io
import json
import math
import re
import urllib.request

import pandas as pd

TEMPLATE_FILE = "index.html"
OUTPUT_FILE = "index.html"
DATA_BASE_URL = "https://stats.tennismylife.org/data/{year}.csv"
ONGOING_URL = "https://stats.tennismylife.org/data/ongoing_tourneys.csv"
YEARS_HISTORICAL = [2023, 2024, 2025]  # traktowane jako jeden bucket "2023-2025"
YEAR_CURRENT = 2026                    # osobny bucket (aktualny sezon)


# ----------------------------------------------------------------------
# 1. POBRANIE DANYCH (roczniki + trwające turnieje)
# ----------------------------------------------------------------------
def download_years(years):
    frames = []
    
    # Pobieranie danych z poszczególnych lat
    for year in years:
        url = DATA_BASE_URL.format(year=year)
        print(f"Pobieram {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req) as resp:
                raw_bytes = resp.read()
            df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
            df["season"] = year
            frames.append(df)
            print(f"  Wczytano {year}.csv: {len(df)} meczow")
        except Exception as e:
            print(f"  UWAGA: nie udalo sie pobrac {year}.csv ({e}), pomijam.")
            continue

    # Pobieranie danych z trwających turniejów (ongoing_tourneys.csv)
    print(f"Pobieram {ONGOING_URL} ...")
    req_ongoing = urllib.request.Request(ONGOING_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req_ongoing) as resp:
            raw_bytes = resp.read()
        df_ongoing = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
        if not df_ongoing.empty:
            df_ongoing["season"] = YEAR_CURRENT
            frames.append(df_ongoing)
            print(f"  Wczytano ongoing_tourneys.csv: {len(df_ongoing)} meczow")
    except Exception as e:
        print(f"  UWAGA: nie udalo sie pobrac ongoing_tourneys.csv ({e}), pomijam.")

    if not frames:
        raise SystemExit("BLAD: Nie udalo sie pobrac zadnych danych meczowych.")

    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# 2. WSPOLCZYNNIK ASOWALNOSCI (ACE_DATA) - z podzialem na okresy
# ----------------------------------------------------------------------
def compute_ace_data(matches, year_label):
    needed = ["surface", "w_ace", "l_ace", "w_SvGms", "l_SvGms"]
    m = matches.dropna(subset=needed)

    rows = []
    for _, r in m.iterrows():
        rows.append({"server_id": r["winner_id"], "server_name": r["winner_name"],
                      "returner_id": r["loser_id"], "returner_name": r["loser_name"],
                      "surface": r["surface"], "aces": r["w_ace"], "sv_gms": r["w_SvGms"]})
        rows.append({"server_id": r["loser_id"], "server_name": r["loser_name"],
                      "returner_id": r["winner_id"], "returner_name": r["winner_name"],
                      "surface": r["surface"], "aces": r["l_ace"], "sv_gms": r["l_SvGms"]})

    long_df = pd.DataFrame(rows)
    long_df["aces_per_game"] = long_df["aces"] / long_df["sv_gms"].replace(0, pd.NA)
    long_df = long_df.dropna(subset=["aces_per_game"])

    base_rates = (long_df.groupby(["server_id", "surface"])["aces_per_game"]
                  .mean().reset_index().rename(columns={"aces_per_game": "base_rate"}))

    merged = long_df.merge(base_rates[["server_id", "surface", "base_rate"]],
                            on=["server_id", "surface"], how="left")
    merged = merged[merged["base_rate"] > 0].copy()
    merged["match_ratio"] = merged["aces_per_game"] / merged["base_rate"]

    k = 10.0
    agg = (merged.groupby(["returner_id", "returner_name", "surface"])
           .agg(raw_coefficient=("match_ratio", "mean"), n_matches=("match_ratio", "count"))
           .reset_index())
    agg["ace_suppression_coefficient"] = (
        (agg["n_matches"] / (agg["n_matches"] + k)) * agg["raw_coefficient"]
        + (k / (agg["n_matches"] + k)) * 1.0
    ).round(3)
    agg["low_confidence"] = agg["n_matches"] < 5
    agg["raw_coefficient"] = agg["raw_coefficient"].round(3)

    agg = agg.merge(
        base_rates.rename(columns={"server_id": "returner_id", "server_name": "returner_name"})[
            ["returner_id", "surface", "base_rate"]],
        on=["returner_id", "surface"], how="left"
    )
    agg["base_rate"] = agg["base_rate"].round(2)
    agg["year"] = year_label

    return agg[["returner_id", "returner_name", "surface", "raw_coefficient", "n_matches",
                "ace_suppression_coefficient", "low_confidence", "base_rate", "year"]].to_dict(orient="records")


# ----------------------------------------------------------------------
# 3. H2H_DATA
# ----------------------------------------------------------------------
def compute_h2h_data(matches):
    needed = ["winner_name", "loser_name", "surface", "score", "tourney_date", "tourney_name", "round"]
    h2h = matches.dropna(subset=needed)
    records = h2h[["tourney_date", "tourney_name", "surface", "round", "winner_name",
                    "loser_name", "score", "w_ace", "l_ace", "w_df", "l_df"]].rename(
        columns={"tourney_date": "date", "tourney_name": "tournament", "winner_name": "winner",
                 "loser_name": "loser", "w_ace": "w_aces", "l_ace": "l_aces",
                 "w_df": "w_dfs", "l_df": "l_dfs"}
    )
    records["date"] = (records["date"].astype(str).str.slice(0, 4) + "-" +
                        records["date"].astype(str).str.slice(4, 6) + "-" +
                        records["date"].astype(str).str.slice(6, 8))
    records = records.sort_values("date", ascending=False)

    out = records.to_dict(orient="records")
    for r in out:
        for key in ("w_aces", "l_aces", "w_dfs", "l_dfs"):
            v = r.get(key)
            if isinstance(v, float) and math.isnan(v):
                r[key] = None
    return out


# ----------------------------------------------------------------------
# 4. EXTRA_STATS (BP converted %, tie-break win rate)
# ----------------------------------------------------------------------
def compute_extra_stats(matches):
    needed_bp = ["surface", "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
    bp_matches = matches.dropna(subset=needed_bp)

    bp_rows = []
    for _, m in bp_matches.iterrows():
        bp_rows.append({"player_id": m["winner_id"], "player_name": m["winner_name"], "surface": m["surface"],
                         "bp_conv": m["l_bpFaced"] - m["l_bpSaved"], "bp_opp": m["l_bpFaced"]})
        bp_rows.append({"player_id": m["loser_id"], "player_name": m["loser_name"], "surface": m["surface"],
                         "bp_conv": m["w_bpFaced"] - m["w_bpSaved"], "bp_opp": m["w_bpFaced"]})

    bp_df = pd.DataFrame(bp_rows)
    bp_agg = bp_df.groupby(["player_id", "player_name", "surface"]).agg(
        bp_conv=("bp_conv", "sum"), bp_opp=("bp_opp", "sum"), n_matches=("bp_opp", "count")
    ).reset_index()
    bp_agg = bp_agg[bp_agg["bp_opp"] > 0].copy()
    bp_agg["bp_converted_pct"] = (bp_agg["bp_conv"] / bp_agg["bp_opp"] * 100).round(1)
    bp_records = bp_agg[["player_id", "player_name", "surface", "bp_converted_pct", "bp_opp", "n_matches"]].rename(
        columns={"bp_opp": "bp_opportunities"}).to_dict(orient="records")

    tb_pattern = re.compile(r"(\d+)-(\d+)\((\d+)\)")
    tb_counts = {}

    def add_tb(name, played, won):
        if name not in tb_counts:
            tb_counts[name] = {"played": 0, "won": 0}
        tb_counts[name]["played"] += played
        tb_counts[name]["won"] += won

    score_matches = matches.dropna(subset=["score", "winner_name", "loser_name"])
    for _, m in score_matches.iterrows():
        for s in str(m["score"]).split(" "):
            match = tb_pattern.search(s)
            if not match:
                continue
            g1, g2 = int(match.group(1)), int(match.group(2))
            if g1 == g2:
                continue
            winner_won_tb = g1 > g2
            add_tb(m["winner_name"], 1, 1 if winner_won_tb else 0)
            add_tb(m["loser_name"], 1, 0 if winner_won_tb else 1)

    tb_records = [
        {"player_name": name, "tb_played": c["played"], "tb_won": c["won"],
         "tb_win_pct": round(c["won"] / c["played"] * 100, 1)}
        for name, c in tb_counts.items() if c["played"] > 0
    ]

    return {"bp": bp_records, "tb": tb_records}


# ----------------------------------------------------------------------
# 5. MATCH_ACES_STATS (srednia/wariancja asow na mecz, wg formatu bo3/bo5)
# ----------------------------------------------------------------------
def compute_match_aces_stats(matches):
    needed = ["surface", "w_ace", "l_ace", "best_of"]
    m = matches.dropna(subset=needed)

    rows = []
    for _, r in m.iterrows():
        rows.append({"player_name": r["winner_name"], "surface": r["surface"],
                      "best_of": int(r["best_of"]), "aces": r["w_ace"]})
        rows.append({"player_name": r["loser_name"], "surface": r["surface"],
                      "best_of": int(r["best_of"]), "aces": r["l_ace"]})

    df = pd.DataFrame(rows)
    agg = df.groupby(["player_name", "surface", "best_of"])["aces"].agg(["mean", "var", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "mean_aces", "var": "var_aces", "count": "n_matches"})
    agg = agg[agg["n_matches"] >= 2].copy()
    agg["var_aces"] = agg["var_aces"].fillna(agg["mean_aces"])
    agg["mean_aces"] = agg["mean_aces"].round(2)
    agg["var_aces"] = agg["var_aces"].round(2)

    return agg.to_dict(orient="records")


# ----------------------------------------------------------------------
# GLOWNY PRZEBIEG
# ----------------------------------------------------------------------
def main():
    all_matches = download_years(YEARS_HISTORICAL + [YEAR_CURRENT])
    historical = all_matches[all_matches["season"].isin(YEARS_HISTORICAL)]
    current = all_matches[all_matches["season"] == YEAR_CURRENT]

    print("\nLiczenie wspolczynnika asowalnosci...")
    ace_data = compute_ace_data(historical, "2023-2025") + compute_ace_data(current, "2026")

    print("Liczenie danych H2H...")
    h2h_data = compute_h2h_data(all_matches)

    print("Liczenie BP converted% i tie-breakow...")
    extra_stats = compute_extra_stats(all_matches)

    print("Liczenie sredniej/wariancji asow na mecz...")
    match_aces_stats = compute_match_aces_stats(all_matches)

    print(f"\nWczytuje szablon: {TEMPLATE_FILE}")
    with open(TEMPLATE_FILE, encoding="utf-8") as f:
        html = f.read()

    def replace_block(html, var_name, new_value_json):
        match = re.search(r"var\s+" + re.escape(var_name) + r"\s*=\s*", html)
        if not match:
            raise SystemExit(
                f"\nBLAD: nie znaleziono deklaracji 'var {var_name} = ' w pliku {TEMPLATE_FILE}.\n"
                f"To zwykle oznacza, ze plik zostal uszkodzony/zmieniony.\n"
            )
        start = match.start()
        marker = match.group(0)
        end = html.index(";", match.end()) + 1
        return html[:start] + marker + new_value_json + ";" + html[end:]

    html = replace_block(html, "ACE_DATA", json.dumps(ace_data, ensure_ascii=False, separators=(",", ":")))
    html = replace_block(html, "H2H_DATA", json.dumps(h2h_data, ensure_ascii=False, separators=(",", ":")))
    html = replace_block(html, "EXTRA_STATS", json.dumps(extra_stats, ensure_ascii=False, separators=(",", ":")))
    html = replace_block(html, "MATCH_ACES_STATS", json.dumps(match_aces_stats, ensure_ascii=False, separators=(",", ":")))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nGotowe. Zapisano: {OUTPUT_FILE} ({len(html):,} znakow)")


if __name__ == "__main__":
    main()
