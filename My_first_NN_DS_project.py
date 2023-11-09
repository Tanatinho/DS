import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests

url = 'https://people.sc.fsu.edu/~jburkardt/data/csv/hw_25000.csv'
res = requests.get(url, allow_redirects=True)
with open('hw_25000.csv','wb') as file:
    file.write(res.content)
df = pd.read_csv('hw_25000.csv')

# But it's also possible just like:
# df = pd.read_csv("https://people.sc.fsu.edu/~jburkardt/data/csv/hw_25000.csv")

print(df.columns)
# We print headers of the columns. As we can see there are quotes and spaces. Remove quotes.
print("As we can see there are quotes and spaces. Let's lemove quotes.")
new_headers = []
for header in df.columns: # df.columns is our list of headers
    header = header.replace('"', '') # Remove the quotes off in each header
    new_headers.append(header) # Save the new strings without the quotes
df.columns = new_headers # Replace the old headers with the new list
print(df.columns)
# As we can see, quotes have been removed but there are space. Let's remove them.
print("Now, let's remove spaces in the headers.")
df.columns = df.columns.str.strip()
df.columns.tolist()
print(df.columns)
# Quotes and spaces have been removed
print("Now, there are no quotes, no spaces in the headers")
print(df)
# Prints heads as wel as the first five and the last five rows.
print("These are the first five and the last five rows.")
print(df.index) # The range is from 0 to 25000 with the step = 1
print(df.head()) # Prints the first five rows
print(df.tail()) # Prints the last five rows
print(df.sort_values(["Height(Inches)"], ascending=False).head) # Prints the data sorted by the decreasing height
print(df.sort_values(["Weight(Pounds)"], ascending=False).head) # Prints the data sorted by the decreasing weight
print(df.info()) # Prints the general information about the data
print(df.describe()) # Prints the general statistic about the data.
# The mean height is 67.993114 inches. The mean weight is 127.079421 pounds.
# The minimal height is 60.278360 inches, the minima weight is 78.014760 pounds.
# The maximal height is 75.152800inches, the maximal weight is 170.924000 pounds.
print("Mode:", df.loc[:, "Height(Inches)"].mode()) # The most frequently appeared height
print("Mode:", df.loc[:, "Weight(Pounds)"].mode())# The most frequently appeared weight
print(df.iloc[2,2]) # Prints the value in the third row and the third column
# Now we show the diagrams of the distribution of the laborers by the weight and by the height
# to see the second diagramm, please, close the first one
plt.hist(df["Weight(Pounds)"], bins = 25, label = "Weight")
plt.title("Распределение сотрудников по весам")
plt.xlabel("Вес, в фунтах")
plt.ylabel("Количество cотрудников")
plt.legend()
plt.show()
plt.hist(df["Height(Inches)"], bins = 25, label = "Height")
plt.title("Распределение сотрудников по росту")
plt.xlabel("Рост, в дюймах")
plt.ylabel("Количество cотрудников")
plt.legend()
plt.show()
