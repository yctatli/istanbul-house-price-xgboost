"""
ADIM 5 - DEĞERLENDİRME VE YORUMLAMA
Kullanım:  python degerlendirme.py
    model/xgb_model.json, model/ayrim.json ve ozellik_data/ dosyalarını okur.
Çıktılar (degerlendirme/ klasörüne yazılır):
    degerlendirme_raporu.txt ve grafikler
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KOK = Path(__file__).resolve().parent
CIKTI = KOK / "degerlendirme"
CIKTI.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
rapor_satirlari = []


def rapor(metin=""):
    print(metin)
    rapor_satirlari.append(str(metin))


def baslik(t):
    rapor("\n" + "=" * 75 + f"\n{t}\n" + "=" * 75)


def kaydet(ad):
    plt.tight_layout()
    plt.savefig(CIKTI / ad, dpi=120)
    plt.close()


# ================================================================ Yükleme
liste = json.loads((KOK / "ozellik_data" / "ozellik_listesi.json").read_text(encoding="utf-8"))
OZELLIKLER = liste["sayisal"] + liste["kategorik"]
df = pd.read_csv(KOK / "ozellik_data" / "ozellikler.csv", encoding="utf-8-sig", dtype={"ofis": str})
for kol in liste["kategorik"]:
    df[kol] = df[kol].astype("category")
ayrim = json.loads((KOK / "model" / "ayrim.json").read_text())
test = df.loc[ayrim["test_idx"]].copy()

model = xgb.XGBRegressor()
model.load_model(KOK / "model" / "xgb_model.json")
AGAC = model.best_iteration + 1  # early stopping'in seçtiği ağaç sayısı

test["tahmin"] = np.exp(model.predict(test[OZELLIKLER]))
test["hata_%"] = (test["tahmin"] / test["price"] - 1) * 100       # + = model fazla söyledi
test["mutlak_hata_%"] = test["hata_%"].abs()
rapor(f"Test seti: {len(test):,} ilan | Model: {AGAC} ağaç")
rapor(f"Medyan mutlak hata: %{test['mutlak_hata_%'].median():.1f}")

# SHAP katkıları: her tahmin = başlangıç değeri + her özelliğin katkısı (log ölçeğinde)
dm = xgb.DMatrix(test[OZELLIKLER], enable_categorical=True)
katki = model.get_booster().predict(dm, pred_contribs=True, iteration_range=(0, AGAC))
shap = pd.DataFrame(katki[:, :-1], columns=OZELLIKLER, index=test.index)
baslangic = katki[0, -1]

# ================================================================ 5.1 Özellik önemi
baslik("5.1 ÖZELLİK ÖNEMİ: model en çok neye bakıyor?")
rapor("ortalama_etki_% : bu özellik, tahmini ortalama olarak yüzde kaç yukarı/aşağı itiyor (SHAP)")
rapor("gain_payi_%     : bu özelliğin kullanıldığı bölmelerin toplam kazançtaki payı\n")
onem = pd.DataFrame({
    "ortalama_etki_%": (np.exp(shap.abs()) - 1).mean() * 100,
    "gain_payi_%": pd.Series(model.get_booster().get_score(importance_type="total_gain")),
}).fillna(0)
onem["gain_payi_%"] = onem["gain_payi_%"] / onem["gain_payi_%"].sum() * 100
onem = onem.sort_values("ortalama_etki_%", ascending=False)
rapor(onem.round(1).to_string())

onem["ortalama_etki_%"].head(20)[::-1].plot.barh(figsize=(8, 7), title="Özellik önemi (ortalama etki, %)")
kaydet("01_ozellik_onemi.png")

# ================================================================ 5.2 Model ne öğrendi?
baslik("5.2 MODEL NE ÖĞRENDİ? (özelliğin değerine göre ortalama etkisi, %)")
rapor("Analizde (2. adım) gördüğümüz ilişkileri model de öğrenmiş mi?\n")
gruplar = {
    "building_age": [-1, 0, 5, 10, 20, 27, 35, 50, 200],
    "net_sqm": [0, 50, 75, 100, 150, 200, 300, 1000],
    "kat_orani": [-10, -0.01, 0.01, 0.5, 0.99, 1.0],
    "aidat_m2": [0, 10, 20, 40, 80, 10_000],
}
fig, eks = plt.subplots(2, 2, figsize=(13, 8))
for (kol, sinir), ax in zip(gruplar.items(), eks.flat):
    etki = (np.exp(shap[kol]) - 1) * 100
    t = etki.groupby(pd.cut(test[kol], sinir), observed=True).agg(["size", "mean"])
    if test[kol].isna().any():
        t.loc["(eksik)"] = [test[kol].isna().sum(), etki[test[kol].isna()].mean()]
    rapor(f"--- {kol} ---\n{t.round(1).to_string()}\n")
    ax.scatter(test[kol], etki, s=3, alpha=0.3)
    ax.axhline(0, color="red", lw=1)
    ax.set_title(f"{kol}: tahmine etkisi (%)")
    if kol in ("net_sqm", "aidat_m2"):
        ax.set_xscale("log")
kaydet("02_model_ne_ogrendi.png")

for kol in ["deed_status", "heating_type", "building_condition", "credit_eligible"]:
    etki = (np.exp(shap[kol]) - 1) * 100
    t = etki.groupby(test[kol], observed=True).agg(["size", "mean"]).query("size >= 10")
    rapor(f"--- {kol} ---\n{t.sort_values('mean').round(1).to_string()}\n")

# ================================================================ 5.3 Tek bir tahminin açıklaması
baslik("5.3 TEK BİR TAHMİNİN ANATOMİSİ")
rapor("Ev örneğini hatırlayın: tahmin = başlangıç + ağaçların düzeltmeleri.")
rapor("SHAP, 562 ağacın toplam düzeltmesini özelliklere paylaştırır.\n")
ornek_idx = (test["mutlak_hata_%"] - test["mutlak_hata_%"].median()).abs().idxmin()  # tipik bir ilan
o = test.loc[ornek_idx]
rapor(f"İlan {o['listing_id']} | {o['mahalle']} | {o['net_sqm']:.0f} m² | yaş {o['building_age']:.0f} | "
      f"{o['deed_status']}")
rapor(f"Başlangıç (tüm evlerin ortalaması): {np.exp(baslangic) / 1e6:6.2f} M TL")
deger = np.exp(baslangic)
sira = shap.loc[ornek_idx].sort_values(key=abs, ascending=False)
for kol, k in sira.head(8).items():
    deger *= np.exp(k)
    rapor(f"  {kol:<20} = {str(o[kol])[:25]:<25} {(np.exp(k) - 1) * 100:+6.1f}%  -> {deger / 1e6:6.2f} M TL")
kalan = sira.iloc[8:].sum()
rapor(f"  {'diğer ' + str(len(sira) - 8) + ' özellik':<46} {(np.exp(kalan) - 1) * 100:+6.1f}%  "
      f"-> {o['tahmin'] / 1e6:6.2f} M TL")
rapor(f"Tahmin: {o['tahmin'] / 1e6:.2f} M TL | Gerçek: {o['price'] / 1e6:.2f} M TL | Hata: %{o['hata_%']:+.1f}")

# ================================================================ 5.4 Model nerede zorlanıyor?
baslik("5.4 MODEL NEREDE ZORLANIYOR?")
rapor("medyan_hata_%   : tipik hata büyüklüğü")
rapor("yon_%           : + ise model sistematik olarak FAZLA, - ise EKSİK tahmin ediyor\n")


def hata_tablosu(grup):
    return test.groupby(grup, observed=True).agg(
        ilan=("price", "size"),
        medyan_hata_=("mutlak_hata_%", "median"),
        yon_=("hata_%", "median"),
    ).rename(columns={"medyan_hata_": "medyan_hata_%", "yon_": "yon_%"})


fiyat_dilim = pd.qcut(test["price"], 10, labels=[f"{i * 10}-{i * 10 + 10}%" for i in range(10)])
t = hata_tablosu(fiyat_dilim)
t.insert(1, "fiyat_araligi_M", test.groupby(fiyat_dilim, observed=True)["price"].agg(
    lambda s: f"{s.min() / 1e6:.1f}-{s.max() / 1e6:.1f}"))
rapor("--- Fiyat dilimlerine göre (en ucuz %10'dan en pahalı %10'a) ---")
rapor(t.round(1).to_string())

t = hata_tablosu("district").query("ilan >= 30").sort_values("medyan_hata_%")
rapor("\n--- İlçelere göre (en az 30 test ilanı olanlar) ---")
rapor(t.round(1).to_string())
t["medyan_hata_%"].plot.barh(figsize=(8, 9), title="İlçelere göre medyan hata (%)")
kaydet("03_ilce_hata.png")

plt.figure(figsize=(6.5, 6))
plt.scatter(test["price"] / 1e6, test["tahmin"] / 1e6, s=3, alpha=0.3)
s = [test["price"].min() / 1e6, test["price"].max() / 1e6]
plt.plot(s, s, "r--", lw=1)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Gerçek fiyat (M TL)"); plt.ylabel("Tahmin (M TL)")
plt.title("Tahmin ve gerçek (test seti)")
kaydet("04_tahmin_gercek.png")

# ================================================================ 5.5 En büyük hatalar
baslik("5.5 EN BÜYÜK HATALAR: sorun modelde mi, ilanda mı?")
goster = ["listing_id", "mahalle", "net_sqm", "rooms", "building_age", "deed_status", "supheli_m2",
          "price", "tahmin", "hata_%"]
fmt = test[goster].copy()
fmt["price"] = (fmt["price"] / 1e6).round(2)
fmt["tahmin"] = (fmt["tahmin"] / 1e6).round(2)
fmt["hata_%"] = fmt["hata_%"].round(0)
rapor("--- Modelin çok FAZLA tahmin ettiği 10 ilan (ilan fiyatı 'ucuz' görünüyor) ---")
rapor(fmt.sort_values("hata_%", ascending=False).head(10).to_string(index=False))
rapor("\n--- Modelin çok EKSİK tahmin ettiği 10 ilan (ilan fiyatı 'pahalı' görünüyor) ---")
rapor(fmt.sort_values("hata_%").head(10).to_string(index=False))
rapor(f"\n%50'den fazla yanılınan ilan: {(test['mutlak_hata_%'] > 50).sum()} "
      f"(%{(test['mutlak_hata_%'] > 50).mean() * 100:.1f})")
rapor(f"Bunların içinde supheli_m2 işaretli olanlar: "
      f"{test.loc[test['mutlak_hata_%'] > 50, 'supheli_m2'].sum():.0f}")

# ================================================================ Kayıt
test.to_csv(CIKTI / "test_detay.csv", index=False)
with open(CIKTI / "degerlendirme_raporu.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rapor_satirlari))
print(f"\nKaydedildi: {CIKTI}/degerlendirme_raporu.txt ve grafikler")