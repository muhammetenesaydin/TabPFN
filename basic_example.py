from tabpfn import TabPFNClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
X, y = load_iris(return_X_y=True)

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TABPFN modelini oluştur
model = TabPFNClassifier(device='cpu')  # 'cuda' yazarsan GPU kullanır

# Eğitimi başlat
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Başarı oranı
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
