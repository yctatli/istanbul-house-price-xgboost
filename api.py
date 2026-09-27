"""
EV FİYAT TAHMİN API'Sİ
Kullanım:  ./venv/bin/uvicorn api:app --reload
    Arayüz (Swagger):  http://127.0.0.1:8000/docs

Uç noktalar:
    POST /tahmin      -> ilan bilgilerini alır, tahmini fiyatı döner
    GET  /secenekler  -> kategorik alanlar için modelin tanıdığı değerler
    GET  /saglik      -> model yüklü mü, hangi ayarla eğitildi
"""
from typing import Literal, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from src.tahmin import Tahminci

app = FastAPI(title="İstanbul Ev Fiyat Tahmini", version="1.0")
tahminci = Tahminci()


class Ilan(BaseModel):
    district: str = Field(..., description="İlçe", examples=["Kadıköy"])
    neighborhood: Optional[str] = Field(None, description="Mahalle", examples=["Caferağa"])
    net_sqm: float = Field(..., gt=0, description="Net m²", examples=[100])
    gross_sqm: Optional[float] = Field(None, gt=0, description="Brüt m²", examples=[120])
    rooms: float = Field(..., ge=0, description="Oda sayısı (3+1 için 3)", examples=[3])
    halls: float = Field(1, ge=0, description="Salon sayısı (3+1 için 1)", examples=[1])
    bathroom_count: Optional[float] = Field(None, ge=0, examples=[1])
    floor: Optional[float] = Field(None, description="Bulunduğu kat (giriş=0, bodrum=-1)", examples=[3])
    floor_category: Optional[str] = Field(None, description="Örn. 'Mid Floor', 'Top Floor', 'Garden Floor'")
    total_floors: Optional[float] = Field(None, ge=0, description="Binadaki toplam kat", examples=[6])
    building_age: Optional[float] = Field(None, ge=0, description="Bina yaşı", examples=[10])
    is_in_complex: bool = Field(False, description="Site içinde mi")
    maintenance_fee: Optional[float] = Field(None, ge=0, description="Aylık aidat (TL)")
    orientation: Optional[list[Literal["North", "South", "East", "West"]]] = Field(
        None, description="Cepheler", examples=[["South", "West"]])
    building_condition: Optional[str] = Field(None, description="'New', 'Second-hand', 'Under Construction'")
    building_type: Optional[str] = None
    deed_status: Optional[str] = None
    credit_eligible: Optional[str] = Field(None, description="'Eligible' / 'Not Eligible'")
    usage_status: Optional[str] = Field(None, description="'Vacant', 'Owner-occupied', 'Tenant-occupied'")
    furnished: Optional[str] = Field(None, description="'Furnished' / 'Unfurnished'")
    heating_type: Optional[str] = Field(None, examples=["Combi Boiler"])
    fuel_type: Optional[str] = Field(None, examples=["Natural Gas"])
    exchange: Optional[str] = Field(None, description="Takasa açık mı: 'Yes' / 'No'")


class TahminCevabi(BaseModel):
    tahmini_fiyat: float = Field(..., description="TL, bine yuvarlanmış")
    aralik_alt: float = Field(..., description="Tahmin -%13 (testteki medyan hata)")
    aralik_ust: float = Field(..., description="Tahmin +%13 (testteki medyan hata)")
    m2_fiyati: float
    uyarilar: list[str] = Field(..., description="Modelin tanımadığı ve başka değerle değiştirilen girdiler")


MEDYAN_HATA = 0.13  # reports/iyilestirme_raporu.txt, test seti


@app.post("/tahmin", response_model=TahminCevabi)
def tahmin(ilan: Ilan):
    sonuc = tahminci.tahmin_et(ilan.model_dump())
    fiyat = sonuc["tahmini_fiyat"]
    return TahminCevabi(tahmini_fiyat=fiyat,
                        aralik_alt=round(fiyat * (1 - MEDYAN_HATA), -3),
                        aralik_ust=round(fiyat * (1 + MEDYAN_HATA), -3),
                        m2_fiyati=round(fiyat / ilan.net_sqm),
                        uyarilar=sonuc["uyarilar"])


@app.get("/secenekler")
def secenekler(ilce: Optional[str] = Query(None, description="Verilirse sadece o ilçenin mahalleleri döner")):
    cevap = {k: v for k, v in tahminci.kategoriler.items() if k != "mahalle"}
    mahalleler = [m for m in tahminci.kategoriler["mahalle"] if ilce is None or m.startswith(f"{ilce} / ")]
    cevap["neighborhood"] = sorted({m.split(" / ", 1)[1] for m in mahalleler} - {"Diğer", "Bilinmiyor"})
    return cevap


@app.get("/saglik")
def saglik():
    return {"durum": "hazır", "ayar": tahminci.ayarlar["ayar"],
            "agac_sayisi": tahminci.ayarlar["parametreler"]["n_estimators"]}
