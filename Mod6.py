import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv('features_30_sec.csv')
X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']

standard = StandardScaler()
X_scaled = standard.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

algo = RandomForestClassifier(n_estimators=100, random_state=42)
algo.fit(X_train, y_train)
y_pred = algo.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

summary_df = report_df.iloc[:-3, :][['precision', 'recall', 'f1-score']].round(2)

plt.figure()
summary_df.plot(kind='bar')
plt.title('Model Performance by Genre')
plt.ylabel('Score')
plt.savefig('model performance summary.png')
plt.close()

fig, ax = plt.subplots()
ax.axis('off')
table = ax.table(cellText=summary_df.values,colLabels=summary_df.columns,rowLabels=summary_df.index,loc='center',cellLoc='center')
table.scale(1, 1.5)
plt.savefig('model performance table.png')
plt.close()

results = pd.DataFrame({'True Genre': y_test,'Predicted Genre': y_pred})

wrong_predictions = results[results['True Genre'] != results['Predicted Genre']]
confusion_detail = wrong_predictions.groupby(['True Genre', 'Predicted Genre']).size().reset_index(name='Count')

print(confusion_detail)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
