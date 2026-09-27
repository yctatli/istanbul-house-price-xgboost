"""
MODELİ İÇERİDEN İZLEME
Kullanım:  ./venv/bin/streamlit run izle.py

Örnek bir evin XGBoost ağaçlarında hangi yoldan geçtiğini adım adım gösterir ve
base_score + yaprakların toplamının model.predict ile aynı olduğunu doğrular.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from src.yollar import FINAL_MODEL

st.set_page_config(page_title="Model İzleme", page_icon="🌳", layout="wide")

# İzlenecek ev: app.py'deki varsayılan örnekle aynı
ORNEK_EV = {"district": "Kadıköy", "neighborhood": "Caddebostan", "net_sqm": 120.0, "rooms": 3.0, "halls": 1.0,
            "bathroom_count": 1, "floor": 2, "total_floors": 5, "building_age": 25.0,
            "deed_status": "Condominium Title", "is_in_complex": 0}


@st.cache_resource(show_spinner="Model yükleniyor...")
def yukle():
    from src import tahmin
    # Model JSON'unu ağaçları tek tek okuyabilmek için ayrıca açıyoruz
    with open(FINAL_MODEL, encoding="utf-8") as f:
        ogrenici = json.load(f)["learner"]
    return tahmin, ogrenici


tahmin, m = yukle()
model = tahmin._TAHMINCI.model
KATEGORIK = tahmin._TAHMINCI.kategorik
AGAC = model.best_iteration + 1  # early stopping'in seçtiği ağaç sayısı; predict de bu kadarını kullanır

isimler = m["feature_names"]
agaclar = m["gradient_booster"]["model"]["trees"]
base = float(m["learner_model_param"]["base_score"].strip("[]"))
X, uyarilar = tahmin.ozellik_uret(pd.DataFrame([ORNEK_EV]))
satir = X.iloc[0]
# Ağaçlardaki kategorik bölmeler kategori KODLARIYLA tutulur (eğitimdeki sıra)
kod = {k: {v: i for i, v in enumerate(tahmin.KATEGORI_LISTESI[k])} for k in KATEGORIK}


def yuru(t, satirlar=None):
    """Ağacı kökten yaprağa yürür; (yaprak düğüm no, yaprak değeri) döner. satirlar verilirse adımları yazar."""
    n = 0
    while t["left_children"][n] != -1:
        f = isimler[t["split_indices"][n]]
        v = satir[f]
        if pd.isna(v):
            git_sol = bool(t["default_left"][n])
            kosul = f"{f} eksik -> varsayılan yön ({'sol' if git_sol else 'sağ'})"
        elif t["split_type"][n] == 1:  # kategorik bölme: kümedeki değerler SAĞA gider
            i = t["categories_nodes"].index(n)
            s, z = t["categories_segments"][i], t["categories_sizes"][i]
            kume = set(t["categories"][s:s + z])
            icinde = kod[f][v] in kume
            git_sol = not icinde
            ornek = [tahmin.KATEGORI_LISTESI[f][c] for c in sorted(kume)][:3]
            kosul = (f"{f} {{{', '.join(ornek)}{', ...' if len(kume) > 3 else ''}}} ({len(kume)} değer) "
                     f"içinde mi?  '{v}' -> {'EVET' if icinde else 'HAYIR'}")
        else:  # sayısal bölme: XGBoost float32 ile karşılaştırır
            esik = t["split_conditions"][n]
            git_sol = np.float32(v) < np.float32(esik)
            kosul = f"{f} < {esik:.4g} mi?  ({v:.4g}) -> {'EVET' if git_sol else 'HAYIR'}"
        if satirlar is not None:
            satirlar.append(f"if {kosul}")
        n = t["left_children"][n] if git_sol else t["right_children"][n]
    if satirlar is not None:
        satirlar.append(f"YAPRAK: {t['split_conditions'][n]:+.5f}")
    return n, t["split_conditions"][n]


# ================================================================ Başlık ve ev
st.title("🌳 Model izleme: bir ev ağaçlardan nasıl geçiyor?")
st.caption(f"Model: `{FINAL_MODEL.relative_to(FINAL_MODEL.parent.parent)}` · "
           f"kaydedilen ağaç: {len(agaclar)} · kullanılan: {AGAC} (early stopping)")
with st.expander("İzlenen ev ve model özellikleri", expanded=False):
    s1, s2 = st.columns(2)
    s1.json(ORNEK_EV)
    s2.dataframe(satir.astype(str).rename("değer"), height=400)
if uyarilar:
    st.warning("\n".join(f"- {u}" for u in uyarilar))

# ================================================================ Seçilen ağaçlarda yol
st.subheader("Ağaç içinde yol")
secili = st.multiselect("Gösterilecek ağaçlar (1'den başlar)", list(range(1, AGAC + 1)), default=[1, 2, 1001]
                        if AGAC >= 1001 else [1, 2, AGAC])
for no in secili:
    adimlar = []
    yuru(agaclar[no - 1], adimlar)
    st.markdown(f"**Ağaç {no}**")
    st.code("\n".join(adimlar), language="text")

# ================================================================ Toplam = tahmin mi?
st.subheader("Yaprakların toplamı = tahmin mi?")
yapraklar = [yuru(agaclar[i])[1] for i in range(AGAC)]
toplam = base + sum(yapraklar)
tahmin_tl = float(np.exp(model.predict(X))[0])
s1, s2, s3, s4 = st.columns(4)
s1.metric("base_score", f"{base:.4f}", f"exp = {np.exp(base) / 1e6:.2f} M TL", delta_color="off")
s2.metric(f"{AGAC} yaprağın toplamı", f"{sum(yapraklar):+.4f}")
s3.metric("Elle hesap", f"{np.exp(toplam) / 1e6:.3f} M TL", f"log = {toplam:.4f}", delta_color="off")
s4.metric("model.predict", f"{tahmin_tl / 1e6:.3f} M TL")
(st.success if abs(np.exp(toplam) - tahmin_tl) / tahmin_tl < 1e-4 else st.error)(
    f"Fark: {abs(np.exp(toplam) - tahmin_tl):,.0f} TL")

d = [len(a["left_children"]) for a in agaclar[:AGAC]]
st.code(f"ilk 10 ağacın yaprakları: {np.round(yapraklar[:10], 4)}\n"
        f"son 5: {np.round(yapraklar[-5:], 5)}\n"
        f"düğüm sayısı ort/maks: {np.mean(d):.1f} {max(d)}", language="text")

# ================================================================ XGBoost'un kendi yapraklarıyla karşılaştırma
st.subheader("Bizim yürüyüş XGBoost ile aynı yaprağa mı varıyor?")
leaf = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_leaf=True,
                                   iteration_range=(0, AGAC))[0]
fark = [i for i in range(AGAC) if yuru(agaclar[i])[0] != leaf[i]]
gercek = np.exp(base + sum(agaclar[i]["split_conditions"][int(leaf[i])] for i in range(AGAC)))
st.write(f"Uyuşmayan ağaç: **{len(fark)}** {fark[:5] if fark else ''} · "
         f"XGBoost yapraklarıyla toplam: **{gercek / 1e6:.3f} M TL**")
if fark:
    i = fark[0]
    adimlar = []
    yuru(agaclar[i], adimlar)
    st.error(f"İlk fark: ağaç {i + 1}, model yaprağı {int(leaf[i])} "
             f"({agaclar[i]['split_conditions'][int(leaf[i])]:+.5f})")
    st.code("\n".join(adimlar), language="text")
