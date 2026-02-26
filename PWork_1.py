import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

#Часть 1
df = pd.read_csv(r"B:\Учёба\2 курс\4 Семестр\Введение в ИИ\Практическая работа  1\developers_salary.csv")
print("Первые 10 срок")
print(df.head(10))

print("______________________________________________________________________________________________________________")
print("Общая информацию о датасете")
print(df.info())

df = df.dropna()
print("______________________________________________________________________________________________________________")
print("Базовая статистика по числовым признакам")
print(df.describe())

print("______________________________________________________________________________________________________________")
print(f"Максимальный опыт работы: {df['опыт_лет'].max()}")
print(f"Минимальный опыт работы: {df['опыт_лет'].min()}")
print(f"Средняя зарплата: {df['зарплата'].mean()}")


mean = df['зарплата'].mean()
std = df['зарплата'].std()


plt.figure(figsize=(12, 7))


# Находим минимальную и максимальную зарплату
min_salary = df['зарплата'].min()
max_salary = df['зарплата'].max()

# Округляем до ближайших сотен тысяч 
min_bin = (min_salary // 100) * 100  
max_bin = (max_salary // 100 + 1) * 100 

bins = np.arange(min_bin, max_bin + 100, 100)

# Построение гистограммы
plt.hist(df['зарплата'], bins=bins, edgecolor='black', color='skyblue')
plt.title('Гистограмма распределения зарплат', fontsize=14)
plt.xlabel('Зарплата (в тыс. руб.)', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.xticks(bins, rotation=45)
plt.tight_layout()
plt.show()

#Выбросы
df.boxplot(column='зарплата')
plt.title('Boxplot')
plt.show()

#Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['зарплата'], df['опыт_лет'])
plt.title('Scatter Plot', fontsize=14)
plt.xlabel('Зарплата', fontsize=12)
plt.ylabel('Опыт_лет', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

#Часть 2
#Преобразование категориальных признаков
df_encoded = df.copy()
categorical_columns = ['образование', 'город', 'язык_программирования', 'размер_компании', 'английский']
df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True)
df_encoded = df_encoded.astype(float)
print("______________________________________________________________________________________________________________")
print("Датасет после One-Hot Encoding")
print(df_encoded.info())

#Корреляционный анализ
print("______________________________________________________________________________________________________________")
print("Топ 10 коррелируемых признаков")
salary = 'зарплата'
correlations = df_encoded.corr()[salary] 
correlations = correlations.drop(salary)
correlations = correlations.iloc[correlations.abs().argsort()[::-1]]

top10 = correlations.head(10)
for feature, corrValue in top10.items():
    print(f"{feature:30}: {corrValue}")

#Тепловая карта
corr_matrix = df_encoded.corr()

plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt='.2f', center=0)
plt.title('Тепловая карта')
plt.tight_layout()
plt.show()

#Часть 3
#Разделение
X = df_encoded.drop('зарплата', axis=1)
y = df_encoded['зарплата'] 
X = X.drop('образование_код', axis=1, errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("______________________________________________________________________________________________________________")
print(f"Размер обучающей выборки: {X_train.shape}") 
print(f"Размер тестовой выборки: {X_test.shape}")

#Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)
#Предсказания
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
#вычисление метрик
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("______________________________________________________________________________________________________________")
print("Метрики на обучающей выборке") 
print(f"MSE: {mse_train:.4f}")
print(f"R²: {r2_train:.4f}") 
print(f"RMSE: {rmse_train:.2f} тыс. руб.") 
print(f"MAE: {mae_train:.2f} тыс. руб.")

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("______________________________________________________________________________________________________________")
print("\nМетрики на тестовой выборке") 
print(f"MSE: {mse_test:.4f}")
print(f"R²: {r2_test:.4f}") 
print(f"RMSE: {rmse_test:.2f} тыс. руб.") 
print(f"MAE: {mae_test:.2f} тыс. руб.")

#DataFrame с коэффициентами модели
coefficients_df = pd.DataFrame({
    'Признак': X_train.columns,
    'Коэффициент': model.coef_
})
#Сортировка коэффициентов
coefficients_df['abs_coef'] = np.abs(coefficients_df['Коэффициент'])
coefficients_df = coefficients_df.sort_values('abs_coef', ascending=False)

print("______________________________________________________________________________________________________________")
print("Топ-10 самых важных признаков")
#топ-10 самых важных признаков
top10 = coefficients_df.head(10)
for i, row in top10.iterrows():
    print(f"{row['Признак']}: {row['Коэффициент']:.2f}")

#bar plot
plt.figure(figsize=(10, 10))
plt.barh(range(len(coefficients_df)), coefficients_df['Коэффициент'])
plt.yticks(range(len(coefficients_df)), coefficients_df['Признак'])
plt.xlabel('Значение коэффициента')
plt.title('Bar plot')
plt.show()

#Часть 4
vif_data = pd.DataFrame()
vif_data["Признак"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i)
for i in range(len(X_train.columns))]
print("______________________________________________________________________________________________________________")
print("\nVIF для всех признаков:")
print(vif_data.sort_values("VIF", ascending=False))

high_vif = vif_data[vif_data["VIF"] > 10]
print("\nПризнаки с VIF > 10:")
print(high_vif)


if len(high_vif) > 0:
    max_vif_feature = high_vif.loc[high_vif['VIF'].idxmax(), 'Признак']

    X_train_improved = X_train.drop(max_vif_feature, axis=1)
    X_test_improved = X_test.drop(max_vif_feature, axis=1)
    
    model_improved = LinearRegression()
    model_improved.fit(X_train_improved, y_train)
    
    y_test_pred_improved = model_improved.predict(X_test_improved)
    
    r2_improved = r2_score(y_test, y_test_pred_improved)
    mse_improved = mean_squared_error(y_test, y_test_pred_improved)
    print("______________________________________________________________________________________________________________")
    print(f"\nКачество: ")
    print(f"R²: {r2_improved:.4f} (было: {r2_test:.4f})")
    print(f"MSE: {mse_improved:.4f} (было: {mse_test:.4f})")
else:
    print("\nНет признаков с VIF > 10, модель не улучшаем")

#Часть 5
X_train_sm = sm.add_constant(X_train)
model_stats = sm.OLS(y_train, X_train_sm).fit()
print("______________________________________________________________________________________________________________")
print("\nSummary модели OLS:")
print(model_stats.summary())


significant_features = model_stats.pvalues[1:][model_stats.pvalues[1:] < 0.05].index.tolist()

if significant_features:
    X_train_significant = X_train[significant_features]
    X_test_significant = X_test[significant_features]
    
    model_significant = LinearRegression()
    model_significant.fit(X_train_significant, y_train)
    
    y_test_pred_significant = model_significant.predict(X_test_significant)
    
    r2_significant = r2_score(y_test, y_test_pred_significant)
    mse_significant = mean_squared_error(y_test, y_test_pred_significant)
    mae_significant = mean_absolute_error(y_test, y_test_pred_significant)

    print(f"\nКачество модели на значимых признаках:")
    print(f"R²: {r2_significant:.4f} (было: {r2_test:.4f})")
    print(f"MSE: {mse_significant:.4f} (было: {mse_test:.4f})")
    print(f"MAE: {mae_significant:.4f} (было: {mae_test:.4f})")

#Часть 6

# График 1: Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
plt.xlabel('Реальная зп')
plt.ylabel('Предсказанная зп')
plt.title('Реальные vs Предсказанные значения')
plt.grid(True, alpha=0.3)
plt.show()

# График 2: Residuals plot
plt.figure(figsize=(8, 6))
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals)
plt.axhline(y=0, color='g', lw=2)
plt.xlabel('Предсказаная зп')
plt.ylabel('Остатки')
plt.title('Residuals Plot')
plt.grid(True, alpha=0.3)
plt.show()


#Часть 7
print("\nПрактический кейс")
new_candidate = pd.DataFrame({
    'опыт_лет': [5],
    'возраст': [28]
})

# остальные признаки
for col in X_train.columns:
    if (col not in new_candidate.columns):
        if ('образование' in col and 'Магистр' in col):
            new_candidate[col] = [1]
        elif ('город' in col and 'Москва' in col):
            new_candidate[col] = [1]
        elif ('язык_программирования' in col and 'Python' in col):
            new_candidate[col] = [1]
        elif ('размер_компании' in col and 'Крупная' in col):
            new_candidate[col] = [1]
        elif ('английский' in col and 'B1-B2' in col):
            new_candidate[col] = [1]
        else:
            new_candidate[col] = [0]

predicted_salary = model.predict(new_candidate)
print(f"Предсказанная зарплата: {predicted_salary[0]:.0f} тыс. руб.")