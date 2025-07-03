import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# CSV verisini oku buraya kendi veri setini ekliceksin deneme için veri setini dosyalarda bıraklıyorum. 
df = pd.read_csv("data.csv")

# Özellikler ve hedef değişkeni ayır
X = df.drop(columns=["target"])
y = df["target"]

# Eğer hedef değişken kategorikse encode et
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = TabPFNClassifier(device='cpu')
model.fit(X_train, y_train)

# Tahmin ve değerlendirme
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
