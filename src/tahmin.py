"""
ADIM 7 - TAHMİN (final modeli kullanma)
Kullanım (kod içinden):
    from src.tahmin import Tahminci
    t = Tahminci()
    t.tahmin_et({"district": "Kadıköy", "neighborhood": "Moda", "net_sqm": 100, ...})

Ham ilan bilgilerini (temiz_veri.csv sütunlarıyla aynı adlar) alır, ozellik.py'deki
dönüşümlerin AYNISINI uygular ve models/xgb_final.json ile fiyat tahmini üretir.

ÖNEMLİ: Buradaki özellik hesapları ozellik.py ile birebir aynı olmalı. ozellik.py'de bir
değişiklik yapılırsa bu dosya da güncellenmeli (kontrol: python -m src.tahmin --kontrol).
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.yollar import FINAL_AYARLAR, FINAL_MODEL, OZELLIK_LISTESI, OZELLIKLER, TEMIZ_VERI


KATEGORIK_ALANLAR = ["district", "floor_category", "building_condition", "building_type", "deed_status",
                     "credit_eligible", "usage_status", "furnished", "heating_type", "fuel_type", "exchange"]
EK_KAT_KURTARMA = {"21 and Above": 21, "Sub-level 1": -1, "Sub-level 2": -2, "Sub-level 3": -3,
                   "Basement & Ground": -1}
YONLER = [("North", "kuzey"), ("South", "guney"), ("East", "dogu"), ("West", "bati")]


def _sayi(deger):
    return np.nan if deger is None else float(deger)


class Tahminci:
    def __init__(self):
        liste = json.loads(OZELLIK_LISTESI.read_text(encoding="utf-8"))
        self.sayisal, self.kategorik = liste["sayisal"], liste["kategorik"]
        self.ozellikler = self.sayisal + self.kategorik
        # Kategori listeleri eğitimdekiyle AYNI sırada olmalı (learn.py: astype("category"))
        veri = pd.read_csv(OZELLIKLER, encoding="utf-8-sig",
                           usecols=self.kategorik)
        self.kategoriler = {k: list(veri[k].astype("category").cat.categories) for k in self.kategorik}
        self.en_sik = {k: veri[k].mode()[0] for k in self.kategorik}
        self.model = xgb.XGBRegressor()
        self.model.load_model(FINAL_MODEL)
        self.ayarlar = json.loads(FINAL_AYARLAR.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ ozellik.py ile aynı dönüşümler
    def ozellik_cikar(self, ilan):
        """Tek ilan (dict) -> (1 satırlık özellik tablosu, uyarı listesi)"""
        uyarilar = []
        g = lambda k: ilan.get(k)
        net, brut = _sayi(g("net_sqm")), _sayi(g("gross_sqm"))
        oda, salon = _sayi(g("rooms")), _sayi(g("halls"))
        x = {"net_sqm": net, "gross_sqm": brut, "rooms": oda, "halls": salon,
             "bathroom_count": _sayi(g("bathroom_count"))}
        x["net_brut_orani"] = net / brut if brut else np.nan
        x["oda_basina_m2"] = net / (oda + salon) if (oda + salon) else np.nan

        kat, toplam = _sayi(g("floor")), _sayi(g("total_floors"))
        kat_turu = g("floor_category") or ""
        if np.isnan(kat) and kat_turu in EK_KAT_KURTARMA:
            kat = float(EK_KAT_KURTARMA[kat_turu])
        x["floor"], x["total_floors"] = kat, toplam
        x["kat_orani"] = kat / toplam if toplam else np.nan
        x["en_ust_kat"] = float(kat == toplam or kat_turu in ("Top Floor", "Penthouse"))
        x["yer_alti"] = np.nan if np.isnan(kat) else float(kat < 0)
        x["giris_kati"] = np.nan if np.isnan(kat) else float(kat == 0)

        aidat = _sayi(g("maintenance_fee"))
        x["building_age"] = _sayi(g("building_age"))
        x["is_in_complex"] = float(bool(g("is_in_complex")))
        x["maintenance_fee"] = aidat
        x["aidat_m2"] = aidat / net if net else np.nan

        yon = g("orientation")
        for ing, tr in YONLER:
            x[f"cephe_{tr}"] = np.nan if yon is None else float(ing in yon)
        x["cephe_sayisi"] = np.nan if yon is None else sum(x[f"cephe_{tr}"] for _, tr in YONLER)

        x["son_guncelleme_gun"] = _sayi(g("son_guncelleme_gun"))
        x["supheli_m2"] = float(g("supheli_m2") or 0)
        x["durum_celiskisi"] = float(g("durum_celiskisi") or 0)

        for kol in KATEGORIK_ALANLAR:
            x[kol] = self._kategori(kol, g(kol), uyarilar)
        ilce = g("district")
        mahalle = f"{ilce} / {g('neighborhood') or 'Bilinmiyor'}"
        if mahalle not in self.kategoriler["mahalle"]:
            yedek = f"{ilce} / Diğer"
            if yedek in self.kategoriler["mahalle"]:
                uyarilar.append(f"mahalle '{mahalle}' eğitimde az/hiç görülmedi -> '{yedek}' kullanıldı")
                mahalle = yedek
            else:
                uyarilar.append(f"mahalle '{mahalle}' bilinmiyor -> boş bırakıldı")
                mahalle = None
        x["mahalle"] = mahalle

        df = pd.DataFrame([x])[self.ozellikler]
        for kol in self.kategorik:
            df[kol] = pd.Categorical(df[kol], categories=self.kategoriler[kol])
        return df, uyarilar

    def _kategori(self, kol, deger, uyarilar):
        izinli = self.kategoriler[kol]
        if deger is None:
            if "Bilinmiyor" in izinli:
                return "Bilinmiyor"
            # Eğitimde hiç boş görülmemiş alan (örn. deed_status): en sık değeri kullan
            uyarilar.append(f"{kol} boş -> en sık değer {self.en_sik[kol]!r} kullanıldı")
            return self.en_sik[kol]
        if deger in izinli:
            return deger
        yedek = "Diğer" if "Diğer" in izinli else ("Bilinmiyor" if "Bilinmiyor" in izinli else None)
        uyarilar.append(f"{kol} '{deger}' eğitimde yok -> {yedek!r} kullanıldı")
        return yedek

    # ------------------------------------------------------------ tahmin
    def tahmin_et(self, ilan):
        df, uyarilar = self.ozellik_cikar(ilan)
        fiyat = float(np.exp(self.model.predict(df)[0]))
        return {"tahmini_fiyat": round(fiyat, -3), "uyarilar": uyarilar}


# ================================================================ Fonksiyon arayüzü (app.py kullanır)
# Aşağıdakiler Tahminci'nin üzerine ince sarmalayıcılardır; özellik hesabı YENİDEN YAZILMAZ,
# her satır Tahminci.ozellik_cikar'dan geçer.
_TAHMINCI = Tahminci()
KATEGORI_LISTESI = _TAHMINCI.kategoriler  # {kategorik sütun: eğitimdeki değerler (sıralı)}


def ozellik_uret(ham_df):
    """temiz_veri.csv biçimindeki satırlar -> (model özellikleri X, uyarılar listesi)"""
    kayitlar = ham_df.astype(object).where(ham_df.notna(), None)
    if "son_guncelleme_gun" not in kayitlar and {"scraped_at", "last_updated"} <= set(ham_df.columns):
        gun = (pd.to_datetime(ham_df["scraped_at"]).dt.normalize() - pd.to_datetime(ham_df["last_updated"])).dt.days
        kayitlar["son_guncelleme_gun"] = gun.astype(object).where(gun.notna(), None)
    parcalar, uyarilar = [], []
    for kayit in kayitlar.to_dict("records"):
        x, satir_uyarilari = _TAHMINCI.ozellik_cikar(kayit)
        parcalar.append(x)
        uyarilar += [u for u in satir_uyarilari if u not in uyarilar]
    X = pd.concat(parcalar, ignore_index=True)
    for kol in _TAHMINCI.kategorik:  # concat kategori tipini bozabilir, yeniden kur
        X[kol] = pd.Categorical(X[kol].astype(object), categories=KATEGORI_LISTESI[kol])
    return X, uyarilar


def tahmin_ve_aciklama(X):
    """X -> (tahmin_TL dizisi, SHAP katkıları). Katkılar log ölçeğinde; son sütun başlangıç değeri.
    Her satırda katkıların toplamı = log(tahmin)."""
    model = _TAHMINCI.model
    tahmin = np.exp(model.predict(X))
    katki = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True,
                                        iteration_range=(0, model.best_iteration + 1))
    return tahmin, katki


def _kontrol():
    """Tutarlılık kontrolü; eşleşmeyen ilan sayısını döner."""
    t = _TAHMINCI
    ham = pd.read_csv(TEMIZ_VERI, encoding="utf-8-sig")
    oz = pd.read_csv(OZELLIKLER, encoding="utf-8-sig", dtype={"ofis": str})
    for kol in t.kategorik:
        oz[kol] = pd.Categorical(oz[kol], categories=t.kategoriler[kol])
    ornek = ham.sample(300, random_state=0)
    ham_kayit = ornek.drop(columns=["price"]).astype(object).where(ornek.notna(), None)
    ham_kayit["son_guncelleme_gun"] = (pd.to_datetime(ornek["scraped_at"]).dt.normalize()
                                       - pd.to_datetime(ornek["last_updated"])).dt.days
    ham_kayit["son_guncelleme_gun"] = ham_kayit["son_guncelleme_gun"].astype(object).where(
        ham_kayit["son_guncelleme_gun"].notna(), None)
    bizim = np.array([t.tahmin_et(r)["tahmini_fiyat"] for r in ham_kayit.to_dict("records")])
    beklenen = np.exp(t.model.predict(oz.loc[ornek.index, t.ozellikler])).round(-3)
    fark = np.abs(bizim - beklenen)
    print(f"{len(ornek)} ilan kontrol edildi, eşleşmeyen: {(fark > 1000).sum()}  (en büyük fark {fark.max():,.0f} TL)")
    # tahmin_ve_aciklama da aynı sonucu vermeli (SHAP toplamı = log tahmin)
    X, _ = ozellik_uret(ornek.drop(columns=["price"]))
    fonk, katki = tahmin_ve_aciklama(X)
    fonk_fark = np.abs(fonk.round(-3) - beklenen)
    shap_fark = np.abs(katki.sum(axis=1) - np.log(fonk)).max()
    return (fark > 1000).sum() + (fonk_fark > 1000).sum() + int(shap_fark > 1e-3), shap_fark


if __name__ == "__main__":
    import sys
    hata, shap_fark = _kontrol()
    if "--kontrol" in sys.argv:
        print(f"ozellik_uret/tahmin_ve_aciklama eşleşmesi kontrol edildi, SHAP toplam farkı {shap_fark:.2e}")
        print("BAŞARILI" if hata == 0 else "BAŞARISIZ")
        sys.exit(0 if hata == 0 else 1)
