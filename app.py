import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Veriyi Excel dosyasından oku
df = pd.read_excel('C:\\Users\\Fatih Enes Savcılı\\Desktop\\bitirmwe\\cancer_patient_datasets.xlsx')

# Eksik değerleri görselleştir
sns.heatmap(df.isnull(), cbar=False)

# Hedef değişkeni (etiketler) seç
y = df[['Level']]

# Bağımsız değişkenleri (özellikler) seç ve gereksiz sütunları çıkar
x = df.drop(columns=['Level', 'Patient Id'], axis=1)

# Veri setini eğitim ve test kümelerine bölelim
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=34)

# Eğitim veri setinin ilk beş gözlemi
print(x_train.head())

# Eğitim etiketlerinin ilk beş gözlemi
print(y_train.head())

# Eğitim veri setindeki eksik değerlerin sayısı
print(x_train.isnull().sum())

# Eğitim etiketlerindeki eksik değerlerin sayısı
print(y_train.isnull().sum())

# Eğitim veri setinden eksik değerleri içeren satırları düşür
x_train = x_train.dropna()

# Eğitim etiketlerinden eksik değerleri içeren satırları düşür
y_train = y_train.dropna()

# Eğitim veri setinin şeklini yazdır
print(x_train.shape)

# Eğitim etiketlerinin şeklini yazdır
print(y_train.shape)

# Karar ağacı sınıflandırıcısı oluştur
tree = DecisionTreeClassifier()

# Modeli eğitim verisiyle eğit
model = tree.fit(x_train, y_train)

# Modelin test verisi üzerindeki doğruluğunu hesapla ve yazdır
print(model.score(x_test, y_test))

# Veri setinin 700. satırındaki değerleri yazdır
a = list(df.iloc[700])
print(a[1:24])

# Modeli kullanarak yeni bir girdiye dayalı tahmin yap
prediction = model.predict([[47, 1, 6, 5, 6, 5, 5, 4, 6, 7, 2, 3, 4, 8, 8, 7, 9, 2, 1, 4, 6, 7, 2]])
print(prediction)



from flask import Flask, render_template, request
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64



app = Flask(__name__, template_folder="templates")
# Arayüz için etiketlerin Türkçe çevirileri
etiketler = {
    'Age': 'Yaş',
    'Gender': 'Cinsiyet',
    'Air Pollution': 'Hava Kirliliği',
    'Alcohol use': 'Alkol Kullanımı',
    'Dust Allergy': 'Toz Alerjisi',
    'OccuPational Hazards': 'Mesleki Riskler',
    'Genetic Risk': 'Genetik Risk',
    'chronic Lung Disease': 'Kronik Akciğer Hastalığı',
    'Balanced Diet': 'Dengeli Beslenme',
    'Obesity': 'Obezite',
    'Smoking': 'Sigara İçme',
    'Passive Smoker': 'Pasif İçici',
    'Chest Pain': 'Göğüs Ağrısı',
    'Coughing of Blood': 'Kan Öksürme',
    'Fatigue': 'Yorgunluk',
    'Weight Loss': 'Kilo Kaybı',
    'Shortness of Breath': 'Nefes Darlığı',
    'Wheezing': 'Hırıltı',
    'Swallowing Difficulty': 'Yutma Güçlüğü',
    'Clubbing of Finger Nails': 'Parmak Tırnaklarında Küçülme',
    'Frequent Cold': 'Sık Sık Soğuk Almalar',
    'Dry Cough': 'Kuru Öksürük',
    'Snoring': 'Horlama'
}

# Kullanıcı arayüzünü oluşturalım
ui = {}
for column in df.columns[1:-1]:  # 'Patient ID' ve 'Level' sütunlarını atladık
    if column == 'Gender':
        ui[column] = '<label for="Gender">Cinsiyet:</label><br><select id="Gender" name="Gender"><option value="1">Erkek</option><option value="0">Kadın</option></select><br><br>'
    elif column == 'Age':
        ui[column] = '<label for="Age">Yaş:</label><br><input type="number" id="Age" name="Age" min="18" max="100"><br><br>'
    else:
        ui[column] = f'<label for="{column}">{etiketler[column]}:</label><br><input type="range" id="{column}" name="{column}" min="1" max="10"><br><br>'





def index():
    
   return render_template('index.html', ui=ui)


@app.route('/', methods=['GET','POST'])
def plot():
    if request.method == 'POST':
        values = [int(request.form[f'{column}']) for column in df.columns[1:-1]]
        level = sum(values) / len(values)

        if level < 4:
            level_text = 'Düşük risk'
        elif level < 7:
            level_text = 'Orta derece risk'
        else:
            level_text = 'Yüksek risk'

        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(['Risk Seviyesi'], [level])
        ax.set_title('Risk Seviyesi')
        ax.set_ylabel('Seviye')
        ax.set_xlabel('Risk')

        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        encoded_img = base64.b64encode(output.getvalue()).decode()
        return f'<img src="data:image/png;base64,{encoded_img}"> <p>Verilere Bağlı Tahmini Seviye: {level_text}</p>'
    return render_template('index.html', ui=ui , etiketler=etiketler)


if __name__ == '__main__':
    app.run(debug=True)