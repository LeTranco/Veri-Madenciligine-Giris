import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.preprocessing import PowerTransformer
from sklearn.inspection import permutation_importance

#Veri Setini Yükleme
df = pd.read_csv('grinding.csv')

#Özellikler (X) ve Hedef Sınıf (y)
X = df[['x1', 'x2', 'x3', 'x4', 'T1', 'T2', 'T3']] 
y = df['z2']

#Korelasyon Isı Haritası
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ozellikler Arasi Korelasyon Isi Haritasi')
plt.tight_layout()
plt.savefig('1_korelasyon_matrisi.png')
plt.close()

#Veri Ön İşleme
scaler = PowerTransformer(method='yeo-johnson')
X_scaled = scaler.fit_transform(X)

#Veriyi Eğitim ve Test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Naive Bayes Modelini Eğitme
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

#Cross-Validation
cv_scores = cross_val_score(nb_model, X_scaled, y, cv=5)
print("\n Modelin Genel Tutarliligi")
print(f"Her Katman Icin Basari: {cv_scores}")
print(f"Ortalama Güvenilir Doğruluk: {cv_scores.mean():.4f}")

#Test seti üzerinde tahmin yapma
y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf_matrix.ravel()

sensitivity = tp / (tp + fn) 
specificity = tn / (tn + fp) 

print("\n Temel Degerlendirme Olcutleri")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity:   {specificity:.3f}")

print("\n Detayli Siniflandirma Raporu")
print(classification_report(y_test, y_pred))

#Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'B'], yticklabels=['A', 'B'])
plt.title('Naive Bayes - Confusion Matrix')
plt.xlabel('Tahmin Edilen Sinif')
plt.ylabel('Gercek Sinif')
plt.tight_layout()
plt.savefig('2_confusion_matrix.png')
plt.close()

#ROC Curve ve AUC
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(nb_model, X_test, y_test, name='Naive Bayes')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Eğrisi')
plt.tight_layout()
plt.savefig('3_roc_egrisi.png')
plt.close()

#Permutation Importance
importance_result = permutation_importance(nb_model, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'Ozellik': X.columns,
    'Onem Derecesi': importance_result.importances_mean
}).sort_values(by='Onem Derecesi', ascending=False)

print("\n Ozelliklerin Modele Katki Siralamasi")
print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Onem Derecesi', y='Ozellik', data=importance_df, hue='Ozellik', palette='magma', legend=False)
plt.title('Naive Bayes - Permutation Importance')
plt.xlabel('Modele Katki (Dogruluk Dususu Orani)')
plt.ylabel('Ozellikler (X)')
plt.tight_layout()
plt.savefig('4_ozellik_onem_grafigi.png')