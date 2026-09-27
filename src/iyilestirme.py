"""
ADIM 6 - İYİLEŞTİRME
Kullanım:  python -m src.iyilestirme
    data/features/ ve models/baseline/ayrim.json dosyalarını okur (4. adımla AYNI ayrımı kullanır).
Çıktılar:
    models/xgb_final.json, models/final_ayarlar.json
    reports/iyilestirme_raporu.txt, reports/test_tahminleri_final.csv, reports/supheli_egitim_ilanlari.csv

ALTIN KURAL: Tüm kararlar DOĞRULAMA setine bakılarak verilir.
Test setine yalnızca en sonda, seçilen tek model için BİR KEZ bakılır.
"""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from src.yollar import (AYRIM, BASELINE_MODEL, FINAL_AYARLAR, FINAL_MODEL, MODEL_KLASOR, OZELLIK_LISTESI,
                        RAPOR_KLASOR)
from src.yollar import OZELLIKLER as OZELLIK_CSV

MODEL_KLASOR.mkdir(exist_ok=True)
RAPOR_KLASOR.mkdir(exist_ok=True)
TOHUM = 42
pd.set_option("display.width", 200)
rapor_satirlari = []


def rapor(metin=""):
    print(metin, flush=True)
    rapor_satirlari.append(str(metin))


def baslik(t):
    rapor("\n" + "=" * 75 + f"\n{t}\n" + "=" * 75)


# ================================================================ Veri (4. adımla aynı ayrım)
liste = json.loads(OZELLIK_LISTESI.read_text(encoding="utf-8"))
OZELLIKLER = liste["sayisal"] + liste["kategorik"]
df = pd.read_csv(OZELLIK_CSV, encoding="utf-8-sig", dtype={"ofis": str})
for kol in liste["kategorik"]:
    df[kol] = df[kol].astype("category")
ayrim = json.loads(AYRIM.read_text())
egitim, dogrulama, test = (df.loc[ayrim[k]] for k in ["egitim_idx", "dogrulama_idx", "test_idx"])
rapor(f"Eğitim {len(egitim):,} | Doğrulama {len(dogrulama):,} | Test {len(test):,} (4. adımla aynı)")

TEMEL = dict(n_estimators=5000, learning_rate=0.05, max_depth=6, min_child_weight=3,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, gamma=0.0,
             objective="reg:squarederror")


def egit(parametre, e, d):
    p = {**TEMEL, **parametre}
    model = xgb.XGBRegressor(**p, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                             early_stopping_rounds=100, eval_metric="mae", random_state=TOHUM)
    model.fit(e[OZELLIKLER], e["log_fiyat"], eval_set=[(d[OZELLIKLER], d["log_fiyat"])], verbose=False)
    return model


def olc(gercek, tahmin):
    hata = np.abs(tahmin - gercek) / gercek
    return {"Medyan hata %": np.median(hata) * 100,
            "%10 içinde": (hata <= 0.10).mean() * 100,
            "%20 içinde": (hata <= 0.20).mean() * 100,
            "MAE (M TL)": np.mean(np.abs(tahmin - gercek)) / 1e6}


def degerlendir(model, veri):
    return olc(veri["price"].values, np.exp(model.predict(veri[OZELLIKLER])))


# ================================================================ 6.1 Parametre ayarı
baslik("6.1 PARAMETRE AYARI (sadece doğrulama setiyle)")
rapor("Her ayar eğitimde öğrenir, doğrulamada ölçülür. 'makas' = doğrulama hatası - eğitim hatası")
rapor("(log MAE). Makas büyükse model eğitim verisini ezberliyor demektir.\n")
AYARLAR = {
    "A: 4. adımdaki model":        {},
    "B: daha sığ ağaçlar":         {"max_depth": 4},
    "C: daha derin ağaçlar":       {"max_depth": 8},
    "D: yaprakta daha çok ev":     {"min_child_weight": 15},
    "E: güçlü λ":                  {"reg_lambda": 10.0},
    "F: daha az özellik/ağaç":     {"colsample_bytree": 0.5},
    "G: D + E + F birlikte":       {"min_child_weight": 15, "reg_lambda": 10.0, "colsample_bytree": 0.5},
    "H: mutlak hata kaybı":        {"objective": "reg:absoluteerror"},
    "I: G + mutlak hata kaybı":    {"min_child_weight": 15, "reg_lambda": 10.0, "colsample_bytree": 0.5,
                                    "objective": "reg:absoluteerror"},
}
satirlar = []
for ad, parametre in AYARLAR.items():
    t0 = time.time()
    m = egit(parametre, egitim, dogrulama)
    e_mae = np.mean(np.abs(m.predict(egitim[OZELLIKLER]) - egitim["log_fiyat"]))
    d_mae = np.mean(np.abs(m.predict(dogrulama[OZELLIKLER]) - dogrulama["log_fiyat"]))
    satirlar.append({"ayar": ad, "ağaç": m.best_iteration + 1, "eğitim_mae": e_mae, "doğrulama_mae": d_mae,
                     "makas": d_mae - e_mae, **degerlendir(m, dogrulama), "sn": time.time() - t0})
    rapor(f"  {ad:<28} doğrulama medyan hata %{satirlar[-1]['Medyan hata %']:.2f}  "
          f"({satirlar[-1]['sn']:.0f} sn)")
sonuc = pd.DataFrame(satirlar).set_index("ayar").drop(columns="sn")
rapor("\n" + sonuc.round(3).to_string())
en_iyi_ad = sonuc["Medyan hata %"].idxmin()
EN_IYI = AYARLAR[en_iyi_ad]
rapor(f"\nDoğrulamada en iyi: {en_iyi_ad}")

# ================================================================ 6.2 Hatalı ilanları eğitimden çıkarmak
baslik("6.2 ŞÜPHELİ İLANLARI EĞİTİMDEN ÇIKARMAK")
rapor("Eğitim setindeki her ilan için 'dışarıda bırakılmış' tahmin üretiyoruz (5 katlı, ofise göre):")
rapor("her ilan, kendisini ve ofisini GÖRMEMİŞ bir modelle tahmin edilir. Fiyatı tahminin YARISINDAN")
rapor("düşük olan ilanlar şüphelidir (peşinat fiyatı, yazım hatası vb.) ve eğitimden çıkarılır.")
rapor("TEK YÖNLÜ: tahminden çok PAHALI görünen ilanlar çıkarılmaz. Onlar çoğunlukla modelin bilmediği")
rapor("bir değeri (Boğaz manzarası, yalı, tarihi bina) taşıyan gerçek lüks mülklerdir.")
rapor("Doğrulama ve test setlerine DOKUNULMAZ: model gerçek hayatta bu tür ilanlarla da karşılaşacak.\n")
oof = pd.Series(np.nan, index=egitim.index)
for i, (a, b) in enumerate(GroupKFold(n_splits=5).split(egitim, groups=egitim["ofis"])):
    ic_e, ic_t = egitim.iloc[a], egitim.iloc[b]
    m = egit({**EN_IYI, "n_estimators": 600}, ic_e, dogrulama)
    oof.iloc[b] = m.predict(ic_t[OZELLIKLER])
    rapor(f"  kat {i + 1}/5 tamam")
sapma = oof - egitim["log_fiyat"]          # + ise ilan fiyatı tahminden düşük
SINIR = np.log(2)
supheli = sapma > SINIR
rapor(f"Karşılaştırma için: tahminin 2 katından PAHALI olan (çıkarılmayan) ilan: {(sapma < -SINIR).sum()}")
rapor(f"\nŞüpheli bulunan eğitim ilanı: {supheli.sum()} (%{supheli.mean() * 100:.1f})")
goster = egitim.loc[supheli, ["listing_id", "mahalle", "net_sqm", "building_age", "price"]].copy()
goster["tahmin_M"] = np.exp(oof[supheli]) / 1e6
goster["price"] = goster["price"] / 1e6
goster["kat"] = np.exp(oof[supheli] - egitim.loc[supheli, "log_fiyat"])
rapor("\nÖrnekler (kat: tahmin, ilan fiyatının kaç katı):")
rapor(goster.sort_values("kat", ascending=False).head(10).round(2).to_string(index=False))
goster.to_csv(RAPOR_KLASOR / "supheli_egitim_ilanlari.csv", index=False)

temiz_egitim = egitim[~supheli]
m_temiz = egit(EN_IYI, temiz_egitim, dogrulama)
m_tam = egit(EN_IYI, egitim, dogrulama)
karsi = pd.DataFrame({"Tüm eğitim verisi": degerlendir(m_tam, dogrulama),
                      "Şüpheliler çıkarılmış": degerlendir(m_temiz, dogrulama)}).T
rapor("\nDoğrulama sonuçları (en iyi ayarla):")
rapor(karsi.round(3).to_string())
# İki ölçüde birden kötüleşmiyorsa ve en az birinde iyileşiyorsa temizliği kabul et
fark = karsi.loc["Şüpheliler çıkarılmış"] - karsi.loc["Tüm eğitim verisi"]
temizlik_iyi = (fark["Medyan hata %"] <= 0) and (fark["MAE (M TL)"] <= 0) and \
               (fark["Medyan hata %"] < 0 or fark["MAE (M TL)"] < 0)
rapor(f"Karar: {'şüpheliler çıkarılacak' if temizlik_iyi else 'tüm eğitim verisi kullanılacak'}")

# ================================================================ 6.3 Final: teste BİR KEZ bakış
baslik("6.3 FİNAL MODEL: TEST SETİ (ilk ve tek bakış)")
final = m_temiz if temizlik_iyi else m_tam
eski = xgb.XGBRegressor()
eski.load_model(BASELINE_MODEL)
son = pd.DataFrame({"4. adım modeli": degerlendir(eski, test),
                    f"Final ({en_iyi_ad.split(':')[0]}{' + temizlik' if temizlik_iyi else ''})":
                        degerlendir(final, test)}).T
rapor(son.round(3).to_string())

tahmin = np.exp(final.predict(test[OZELLIKLER]))
yon = (tahmin / test["price"] - 1) * 100
dilim = pd.qcut(test["price"], 5, labels=["en ucuz %20", "20-40", "40-60", "60-80", "en pahalı %20"])
eski_yon = (np.exp(eski.predict(test[OZELLIKLER])) / test["price"] - 1) * 100
rapor("\nOrtalamaya çekilme (medyan sapma yönü, %):")
rapor(pd.DataFrame({"4. adım": eski_yon.groupby(dilim, observed=True).median(),
                    "Final": yon.groupby(dilim, observed=True).median()}).round(1).T.to_string())

final.save_model(FINAL_MODEL)
cikti = test[["listing_id", "mahalle", "net_sqm", "price"]].copy()
cikti["tahmin"] = tahmin.round(-3)
cikti["hata_%"] = yon.round(1)
cikti.to_csv(RAPOR_KLASOR / "test_tahminleri_final.csv", index=False)
json.dump({"ayar": en_iyi_ad, "parametreler": {**TEMEL, **EN_IYI, "n_estimators": final.best_iteration + 1},
           "supheliler_cikarildi": bool(temizlik_iyi)},
          open(FINAL_AYARLAR, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with open(RAPOR_KLASOR / "iyilestirme_raporu.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rapor_satirlari))
print(f"\nKaydedildi: {MODEL_KLASOR}/ ve {RAPOR_KLASOR}/")