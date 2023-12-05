#  Практическое задание № 2. Подготовка данных для анализа. By A. Khodorov

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests

# Set default font size
plt.rcParams['font.size'] = 16

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Splitting data into training and testing
from sklearn.model_selection import train_test_split

# Imputing missing values and scaling values
#from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# Шаг 1. Загружаем данные и создаём DataFrame

url = 'https://data.cityofnewyork.us/api/views/utpj-74fz/rows.csv'
res = requests.get(url, allow_redirects=True)
with open('Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv','wb') as file:
    file.write(res.content)
df = pd.read_csv('Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv', low_memory=False)

print(df)

''' Это данные об энергоэффективности зданий в Нью-Йорке.
Как видим они состоят из 16378 строк и 262 столбцов. Выводим общую информацию о данных.'''

print(df.describe())
print(df.head())

'''  Наша цель: использовать имеющиеся данные для построения модели, которая прогнозирует количество баллов Energy Star Score для конкретного
здания, и интерпретировать результаты для поиска факторов, влияющих на итоговый балл. Т.е. нас, в первую очередь, интрересует колонка
"Energy Star Score". Определим её номер, тип данных и посмотрим её.'''

print('Колонка Energy Star Score имеет номер:', df.columns.get_loc("ENERGY STAR Score"))
print('Тип данных колонки Energy Star Score:', df.dtypes['ENERGY STAR Score'])
print(df[['ENERGY STAR Score']])

# Как видно, в этой колонке, как и в остальной таблице, есть отсутствующие данные («Not Available»).
# Произведём очистку данных. Для начала, сделаем их копию.

df_c = df.copy()
print('Копия данных:', df_c)

''' Если в колонке есть отсуствующие значения, то тип данных для неё будет "object", даже если все имеющиеся данные - численные. Мы не можем
применять к ним числовой анализ. Заменим все «Not Available» на not a number (np.nan), которые можно интерпретировать как числа, а затем
конвертирует содержимое таких колонок в тип float'''

df_c = df_c.replace({'Not Available': np.nan})

for col in list(df_c.columns):
    # Выберем колонки, которые должны иметь численные значения.
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
        col or 'therms' in col or 'gal' in col or 'Score' in col):
        # Конвертируем.
        df_c[col] = df_c[col].astype(float)

# Теперь начнём исследовать данные. Сначала определим отсутствующие данные в каждой колонке.

# Напишем ф-цию для подсчёта отсутсвующих значений в каждом столбце.
def missing_values_table(df):
        # Общее кол-ство отсуств. значений
        mis_val = df_c.isnull().sum()
        
        # Процент отсуств. значений
        mis_val_percent = 100 * df.isnull().sum() / len(df_c)
        
        # Создаём таблицу результатов
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Называем колонки
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Сортируем таблицу по уменьшению % отсуств. значений
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Выводим информацию. Как видим, 236 из 262 столбцов имеют отсуств. значения
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Выводим величины отсуств. значений по столбцам (по убыванию %)
        return mis_val_table_ren_columns

print(missing_values_table(df_c))

# Удалим столбцы, в которых процент отсуств. значений >= 50. Их 164.

# Get the columns with > 50% missing
missing_df_c = missing_values_table(df_c);
missing_columns = list(missing_df_c[missing_df_c['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))

# Drop the columns
df_c = df_c.drop(columns = list(missing_columns))

print(missing_values_table(df_c))

# Как видим, в оставшихся столбцах, % осуств. значений < 50

print(df_c)

# Осталось 98 столбцов. Произведём разведочный анализ данных. Исследуем распределение ENERGY STAR Score

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(df_c['ENERGY STAR Score'].dropna(), bins = 100, edgecolor = 'k')
plt.xlabel('ENERGY STAR Score'); plt.ylabel('Number of Buildings')
plt.title('Energy Star Score Distribution')
plt.show()

''' Как видим, вместо распределения близкого к "плоскому" (Energy Star Score является процентилем), мы видим аномальные величины
для значений 1 и 100. Возможно, это связано с тем, что этот параметр рассчитывается на основе «самостоятельно заполняемых владельцами
зданий отчётов». Теперь посмотрим на параметр Energy Use Intensity (EUI), который определяется как вся использованная
энергия, делённая на общую площадь здания.'''

# Histogram Plot of Site EUI
figsize(8, 8)
plt.hist(df_c['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black')
plt.xlabel('Site EUI') 
plt.ylabel('Count')
plt.title('Site EUI Distribution')
plt.show()

print(df_c['Site EUI (kBtu/ft²)'].describe())

''' Этот параметр не является процентным, поэтому болшое влияние оказывают абсолютные значения. Как видим, огромное значение для одного
из зданий доминирует. Уберём аномальные значения для Energy Use Intensity (EUI) руководствуясь следующими критериями:

Ниже первого квартиля − 3 ∗ интерквартильный размах.
Выше третьего квартиля + 3 ∗ интерквартильный размах.'''

# Удаляем аномальные значения (Removing Outliers)

# Calculate first and third quartile
first_quartile = df_c['Site EUI (kBtu/ft²)'].describe()['25%']
third_quartile = df_c['Site EUI (kBtu/ft²)'].describe()['75%']

# Interquartile range
iqr = third_quartile - first_quartile

# Remove outliers
df_c = df_c[(df_c['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) &
            (df_c['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]

# Histogram Plot of Site EUI
figsize(8, 8)
plt.hist(df_c['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black')
plt.xlabel('Site EUI', size = 14)
plt.ylabel('Count', size = 14); plt.title('Распределние Site EUI', size = 16)
plt.show()

''' Как видно, распределние Count vs Site EUI стало близко к нормальному. Теперь поищем взаимосвязи между признаками и нашей целью.
Коррелирующие с ней переменные полезны для использования в модели, потому что их можно применять для прогнозирования. Один из способов
изучения влияния категориальной переменной (которая принимает только ограниченный набор значений) на цель — это построить график плотности
с помощью библиотеки Seaborn. График плотности можно считать сглаженной гистограммой, потому что он показывает распределение одиночной
переменной. Можно раскрасить отдельные классы на графике, чтобы посмотреть, как категориальная переменная меняет распределение.
Построим график плотности Energy Star Score, раскрашенный в зависимости от типа здания (для списка зданий с более чем 50 измерениями)'''

# Create a list of buildings with more than 50 measurements
types = df_c.dropna(subset=['ENERGY STAR Score'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 50].index)
print(types)

# Это 9 типов зданий: 'Multifamily Housing', 'Office', 'K-12 School', 'Hotel',
#'Non-Refrigerated Warehouse', 'Residence Hall/Dormitory', 'Senior Care Community',
#'Retail Store', 'Distribution Center'

# Plot of distribution of scores for building categories
figsize(12, 9)

# Plot each building
for b_type in types:
    # Select the building type
    subset = df_c[df_c['Largest Property Use Type'] == b_type]
    
    # Density plot of Energy Star Scores
    sns.kdeplot(subset['ENERGY STAR Score'].dropna(),
               label = b_type, fill = False, alpha = 0.8);
    
# label the plot
plt.xlabel('Energy Star Score', size = 14); plt.ylabel('Плотность', size = 14); 
plt.title('График плотности Energy Star Scores в завис. от типа зданий', size = 14);
plt.show()

''' Как видно, тип здания сильно влияет на количество баллов. Офисные здания обычно имеют более высокий балл, а отели более низкий.
Значит нужно включить тип здания в модель, потому что этот признак влияет на нашу цель. Аналогичный график используем для оценки
Energy Star Score по районам города(boroughs):'''

# Plot of distribution of ENERGY STAR Score for boroughs
figsize(12, 10)

# Create a list of boroughs with more than 50 observations
boroughs = df_c.dropna(subset=['ENERGY STAR Score'])
boroughs = boroughs['Borough'].value_counts()
boroughs = list(boroughs[boroughs.values > 50].index)
print(boroughs)

# Таких районов 5: 'MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN IS'

# Plot of distribution of scores for boroughs
figsize(12, 10)

# Plot each borough distribution of scores
for borough in boroughs:
    # Select the building type
    subset = df_c[df_c['Borough'] == borough]
    
    # Density plot of Energy Star scores
    sns.kdeplot(subset['ENERGY STAR Score'].dropna(),
               label = b_type, fill = False, alpha = 0.8)
    
# label the plot
plt.xlabel('Energy Star Score', size = 14); plt.ylabel('Плотность', size = 14)
plt.title('График плотности Energy Star Scores в завис. от района', size = 14)
plt.show()

''' Как видно, район тоже влияет на балл, хотя и не так сильно, как тип здания. Тем не менее, включим его в модель. Для этих категориальных
переменных позже выполним one-hot кодирование. Оценим взаимосвязь Energy Star Scores с другими переменными. Сначала используем для этого
коэффициент корреляции Пирсона. Это мера интенсивности и направления линейной зависимости между двумя переменными. Значение +1 означает
идеально линейную положительную зависимость, а -1 означает идеально линейную отрицательную зависимость.
'''

# Find all correlations with the Energy Star Scores and sort
correlations_df_c = df_c.corr(numeric_only=True)['ENERGY STAR Score'].sort_values()

# Print the most negative correlations
print(correlations_df_c.head(15), '\n')

# Print the most positive correlations
print(correlations_df_c.tail(15))

''' Как видно, значительных положительных корреляций нет, но есть несколько сильных отрицательных корреляций между признаками и
ENERGY STAR Score, причём наибольшие из них относятся к разным категориям EUI (способы расчёта этих показателей слегка различаются).
EUI  (Energy Use Intensity, интенсивность использования энергии) —  это количество энергии, потреблённой зданием, делённое на квадратный
фут площади. Эта удельная величина используется для оценки энергоэффективности, и чем она меньше, тем лучше. Логика подсказывает, что эти
корреляции оправданны: если EUI увеличивается, то Energy Star Score должен снижаться. 
    Для оценки возможных нелинейных корреляций, возьмём квадратный корень и нат. логарифм переменных и рассчитаем их коэф. корреляции с
ENERGY STAR Score. Для оценки возможной корреляции ENERGY STAR Score с районом (Borough) и типом строения, выполним one-hot кодирование
для этих категориальных признаков.
'''

# Select the numeric columns

np.seterr(divide = 'ignore')

numeric_subset = df_c.select_dtypes('number')
print(numeric_subset)
# Create columns with square root and log of numeric columns

np.seterr(divide='ignore')
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'ENERGY STAR Score':
        next
    else:
        with np.errstate(divide='ignore'):
            numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
            numeric_subset['log_' + col] = np.log(numeric_subset[col])

# Select the categorical columns
categorical_subset = df_c[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

# Drop buildings without an energy star score
features = features.dropna(subset = ['ENERGY STAR Score'])

# Find correlations with the score 
correlations = features.corr()['ENERGY STAR Score'].dropna().sort_values()

# Display most negative andpositive correlations
print(correlations.head(15))
print(correlations.tail(15))
#print(categorical_subset)

''' После этих преобразований (логарифмирование и извлечение квадратного корня), наибольшие отрицательные корреляции практически не
изменились, но появились значительные положительные:

log_Avoided Emissions - Offsite Green Power (Metric Tons CO2e)  =  0.528102
log_Green Power - Offsite (kWh)  and  0.550166.

Тип здания, являясь категориальным признаком преобразованным с помощью one-hot encoding, имеет не большую позитивную корреляцию с ENERGY STAR Score.

Позже мы используем эти данные по корреляциям для выбора свойств, которыми будем описывать нашу модель.

А сейчас мы построим Pairs Plot (парный график) для визуализации взаимосвязи между различными парами переменных и распределения одиночных
переменных. Мы воспользуемся библиотекой Seaborn и функцией PairGrid для создания парного графика с диаграммой рассеивания в верхнем
треугольнике, гистограммой по диагонали, двухмерной диаграммой плотности ядра и коэффициентов корреляции в нижнем треугольнике.
'''

# Extract the columns to  plot
plot_data = features[['ENERGY STAR Score', 'Site EUI (kBtu/ft²)', 
                      'Weather Normalized Source EUI (kBtu/ft²)', 
                      'log_Total GHG Emissions (Metric Tons CO2e)']]

# Replace the inf with nan
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})

# Rename columns 
plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI', 
                                        'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI',
                                        'log_Total GHG Emissions (Metric Tons CO2e)': 'log GHG Emissions'})

# Drop na values
plot_data = plot_data.dropna()

# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

# Create the pairgrid object
grid = sns.PairGrid(data = plot_data)

# Upper is a scatter plot
grid.map_upper(plt.scatter, color = 'red', alpha = 0.6)

# Diagonal is a histogram
grid.map_diag(plt.hist, color = 'red', edgecolor = 'black')

# Bottom is correlation and density plot
grid.map_lower(corr_func);
grid.map_lower(sns.kdeplot, cmap = plt.cm.Reds)

# Title for entire plot
plt.suptitle('Pairs Plot of Energy Data', size = 36, y = 1.02)
plt.show()

''' Теперь, когда мы исследовали тенденции и взаимосвязи данных, мы можем выбрать свойства для постороения нашей модели. Мы выберём только
свойства с числовыми значениями и добавим к ним два категориальных признака: район и тип собственности. Числовые значения прологарифмируем,
а для категориальных выполним one-hot кодирование и объединим эти данные.
'''

# Copy the original data
features = df_c.copy()
np.seterr(divide = 'ignore')

# Select the numeric columns
numeric_subset = df_c.select_dtypes('number')

# Create columns with log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'ENERGY STAR Score':
        next
    else:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])
        
# Select the categorical columns
categorical_subset = df_c[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)
print('Esta pronto!')
print(features.shape)

''' Теперь исследуем свойства на коллинеарность. Если какие-либо свойства сильно коррелируют между собой, то они
оказывают примерно одинаковое влияние на нашу цель (ENERGY STAR Score). Можем удалить их, оставив одно, для упрощения модели.
Например, вот график EUI и Weather Normalized Site EUI, у которых коэффициент корреляции равен 0,9967.'''

plot_data = df_c[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna()
plt.plot(plot_data['Site EUI (kBtu/ft²)'], plot_data['Weather Normalized Site EUI (kBtu/ft²)'], 'bo')
plt.xlabel('Site EUI'); plt.ylabel('Weather Norm EUI')
plt.title('Weather Norm EUI vs Site EUI, R = %0.4f' % np.corrcoef(df_c[['Weather Normalized Site EUI (kBtu/ft²)',
            'Site EUI (kBtu/ft²)']].dropna(), rowvar=False)[0][1])
plt.show()

# Удалим признаки с коэффициентом корреляции больше чем 0.6

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between Energy Star Score
    y = x['ENERGY STAR Score']
    x = x.drop(columns = ['ENERGY STAR Score'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    x = x.drop(columns = ['Weather Normalized Site EUI (kBtu/ft²)', 
                          'Water Use (All Water Sources) (kgal)',
                          
                          'Largest Property Use Type - Gross Floor Area (ft²)'])
    
    # Add the score back in to the data
    x['ENERGY STAR Score'] = y
               
    return x

# Remove the collinear features above a specified correlation coefficient
features = remove_collinear_features(features, 0.6)

# Remove any columns with all na values
features  = features.dropna(axis=1, how = 'all')
print('После удаления признаков с коэффициентом корреляции больше чем 0.6', features.shape)

''' Как видим, осталось 15628 строк и 85 столбцов (изначально было 16378 rows x 262 columns). Извлечём строки в которых отсутсвуют
данные для ENERGY STAR Score, поскольку они бесполезны как для обучения, так и для тестирования.'''

# Extract the buildings with no score and the buildings with a score
no_score = features[features['ENERGY STAR Score'].isna()]
score = features[features['ENERGY STAR Score'].notnull()]

print('Пустых строк ENERGY STAR Score: ', no_score.shape)
print('Строк ENERGY STAR Score с данными: ',score.shape)

''' Как видно, пустых строк - 3053. Оставшиеся 12575 разделим случайным образом на обучающий и тестовый наборы
в соотношении 70% и 30%, соответственно, с помощью scikit-learn.'''

# Separate out the features and targets
features = score.drop(columns='ENERGY STAR Score')
targets = pd.DataFrame(score['ENERGY STAR Score'])

# Replace the inf and -inf with nan (required for later imputation)
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

# Split into 70% training and 30% testing set
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)

print('Количество строк для обучения: ', X.shape)
print('Количество строк для тестирования: ', X_test.shape)
print(y.shape)
print(y_test.shape)

''' Итого у нас кол-во зданий для обучения - 8802, для тестирования - 3773.

  Прежде чем перейти к созданию модели, выберим исходный базовый уровень (naive baseline) — некое предположение, с которым мы будем
сравнивать результаты работы моделей. Если они окажутся ниже базового уровня, мы будем считать, что машинное обучение неприменимо
для решения этой задачи, или что нужно попробовать иной подход. Для регрессионных задач в качестве базового уровня разумно угадывать
медианное значение цели на обучающем наборе для всех примеров в тестовом наборе. Эти наборы задают барьер, относительно низкий
для любой модели. В качестве метрики возьмём среднюю абсолютную ошибку (mae) в прогнозах.'''

# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Now we can make the median guess and evaluate it on the test set.

baseline_guess = np.median(y)

print('Предполагаемый базовый уровень: %0.2f' % baseline_guess)
print("Средняя абсолютная ошибка: MAE = %0.4f" % mae(y_test, baseline_guess))

'''The baseline guess is a score of 68.00 Baseline Performance on the test set: MAE = 24.4294. Средняя абсолютная ошибка на тестовом наборе
составила около 25 пунктов. Поскольку мы оцениваем в диапазоне от 1 до 100, то ошибка составляет примерно 25%. Сохраним данные.'''

# Save the no scores, training, and testing data
no_score.to_csv('data/no_score.csv', index = False)
X.to_csv('data/training_features.csv', index = False)
X_test.to_csv('data/testing_features.csv', index = False)
y.to_csv('data/training_labels.csv', index = False)
y_test.to_csv('data/testing_labels.csv', index = False)

'''  Хотя при очистке данных мы отбросили колонки, в которых не хватало больше половины измерений, у нас ещё отсутствует немало значений.
Модели машинного обучения не могут работать с отсутствующими данными, поэтому нам нужно их заполнить. Для этого воспользуемся достаточно
простым методом медианного заполнения (median imputation), который заменяет отсутствующие данные средним значениями по соответствующим
колонкам. С этой целью, создадим Scikit-Learn-объект Imputer с медианной стратегией. Затем обучим его на обучающих данных
(с помощью imputer.fit), и применим для заполнения отсутствующих значений в обучающем и тестовом наборах (с помощью imputer.transform).
То есть записи, которых не хватает в тестовых данных, будут заполняться соответствующим медианным значением из обучающих данных.
Мы делаем заполнение и не обучаем модель на данных как есть, чтобы избежать проблемы с утечкой тестовых данных, когда информация
из тестового датасета переходит в обучающий.
'''

# Read in data into dataframes 
train_features = pd.read_csv('data/training_features.csv')
test_features = pd.read_csv('data/testing_features.csv')
train_labels = pd.read_csv('data/training_labels.csv')
test_labels = pd.read_csv('data/testing_labels.csv')

# Create an imputer object with a median filling strategy
imputer = SimpleImputer(strategy='median')

# Train on the training features
imputer.fit(train_features)

# Transform both training data and testing data
X = imputer.transform(train_features)
X_test = imputer.transform(test_features)
print('Пропущенные значения в тренировочных данных: ', np.sum(np.isnan(X)))
print('Пропущенные значения в тестовых данных:  ', np.sum(np.isnan(X_test)))

''' Теперь все значения заполнены, пропусков нет.
Поскольку признаки измеряются в разных единицах и покрывают разные диапазоны, то это может сильно искажать результаты методов, использующих
различные алгоритмы. Масштабирование позволяет этого избежать. Существуют два основных способа: нормализация (из текущего значения отнимают
минимальное и делят на диапазон (разницу между макс и миним), что приводит к значениям в диапазоне от 0 до 1) и стандартизация
(из текущего значения отнимают среднее и делят на стандартное отклонение). Проведём нормализацию данных, для чего воспользуемся объектом
MinMaxScaler из Scikit-Learn. Параметры нормализации определим на обучающем наборе, а затем преобразуем все данные.
'''

# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)
# Convert y to one-dimensional array (vector)
y = np.array(train_labels).reshape((-1, ))
y_test = np.array(test_labels).reshape((-1, ))
print(X)

#Как видим все значения нормализованы и лежат в дипазоне от 0 до 1.
'''
Таким образом, мы загрузили данные об энергоэффективности зданий в Нью-Йорке и создали на их основе DataFrame.
Оценили состояние данных: вывели информацию о типах данных, наличии пропущенных значений и основных статистических характеристиках.
Очистили и структурировали данные, провели разведочный анализ, собрали набор признаков для использования в модели и установили базовый
уровень для оценки результатов. Преобразовали категориальные признаки (Районы и Тип собственности) в числовые с использованием one-hot encoding.
Разделили данные для обучения и тестирования в соотношении 7:3. Сохранили предобработанные данные. Затем, на основе данных для обучения,
определили параметры для заполнения пропущенных значений методом медианного заполнения и нормализации числовых данных, и преобразовали
все данные, подготовив их для создания/выбора модели обучения.
'''
print('Данные готовы для создания/выбора модели обучения.')
