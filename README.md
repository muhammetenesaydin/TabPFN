# TabPFN
TabPFN clasifier
# Tabular Data Classification with TABPFN and Traditional Models

Bu proje, tabular (tablo) veri sınıflandırma problemlerinde farklı modellerin performansını karşılaştırmak amacıyla hazırlanmıştır. Özellikle yeni nesil **TABPFN** modeli ile klasik yöntemler olan **RandomForest**, **XGBoost** ve **LightGBM** modelleri 5 katlı çapraz doğrulama (cross-validation) ile test edilmiştir.

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)  
- [Kurulum](#kurulum)  
- [Kullanım](#kullanım)  
- [Modeller](#modeller)  
- [Sonuçlar](#sonuçlar)  
- [Kaynaklar](#kaynaklar)  
- [Lisans](#lisans)

---

## Proje Hakkında

TABPFN (Transformer-based Probabilistic Fully-connected Network), tabular veri için geliştirilmiş, önceden eğitilmiş bir Transformer tabanlı sınıflandırıcıdır. Bu model, klasik makine öğrenmesi algoritmalarına göre özellikle küçük ve orta ölçekli veri setlerinde hızlı ve etkili tahminler sunar.

Bu projede, TABPFN modeli ile birlikte RandomForest, XGBoost ve LightGBM modellerinin aynı veri seti üzerinde performansları karşılaştırılmıştır.

---

## Kurulum

Python 3.8 ve üzeri sürümlerde çalışmaktadır.

Gerekli paketleri yüklemek için:

```bash
pip install -r requirements.txt
