# TabPFN
TabPFN clasifier
# Tabular Data Classification with TABPFN and Traditional Models

Bu proje, tabular (tablo) veri sınıflandırma problemlerinde farklı modellerin performansını karşılaştırmak amacıyla hazırlanmıştır. Özellikle yeni nesil **TABPFN** modeli ile klasik yöntemler olan **RandomForest**, **XGBoost** ve **LightGBM** modelleri 5 katlı çapraz doğrulama (cross-validation) ile test edilmiştir.

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)  
- [Kurulum](#kurulum)  
- [Kullanım](#kullanım)  
- [Sonuçlar](#sonuçlar)  
- [Kaynaklar](#kaynaklar)  


---

## Proje Hakkında

TABPFN (Transformer-based Probabilistic Fully-connected Network), tabular veri için geliştirilmiş, önceden eğitilmiş bir Transformer tabanlı sınıflandırıcıdır. Bu model, klasik makine öğrenmesi algoritmalarına göre özellikle küçük ve orta ölçekli veri setlerinde hızlı ve etkili tahminler sunar.

Bu projede, TABPFN modeli ile birlikte RandomForest, XGBoost ve LightGBM modellerinin aynı veri seti üzerinde performansları karşılaştırılmıştır.


## Kurulum
Python 3.8 ve üzeri sürümlerde çalışmaktadır.

---
Gerekli paketleri yüklemek için:

```pip install -r requirements.txt```

---
## Kullanım
Çalıştırmak için:

```basic_example.py```

## Modeller 

## Sonuçlar 
![image](https://github.com/user-attachments/assets/9cbe7243-c756-4d93-b960-1239aec41d3d)
![image](https://github.com/user-attachments/assets/db4c0803-1648-48fc-aa70-150808443bd2)
## Kaynaklar 
🔗 Resmi GitHub Reposu: https://github.com/PriorLabs/TabPFN

🔗 TabPFN Makalesi: https://arxiv.org/abs/2207.01848



