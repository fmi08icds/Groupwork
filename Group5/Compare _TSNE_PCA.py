import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

colors = ['royalblue','red','deeppink', 'maroon', 'mediumorchid', 'tan', 'forestgreen', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

input_path='/kaggle/input/icmr-data/'

df1='data.csv'
df2='labels.csv'

data = pd.read_csv(input_path + df1)
label = pd.read_csv(input_path + df2)

df = pd.merge(label,data)
df.head()

df.isnull().sum()

df.describe()

heatmap_data = pd.pivot_table(df, index=['Class'])

heatmap_data.head()

sns.clustermap(heatmap_data)
plt.savefig('heatmap_with_Seaborn_clustermap_python.jpg',
            dpi=150, figsize=(8,12))

sns.clustermap(heatmap_data, figsize=(18,12))
plt.savefig('clustered_heatmap_with_dendrograms_Seaborn_clustermap_python.jpg',dpi=150)

plt.figure(figsize=(14,6))
plt.hist(df['Class'])
plt.show()

non_cat_data = df.drop(['Unnamed: 0'], axis=1)
non_cat_data

df_f_test=df


def f_test(df_f_test, gene):
    df_anova = df_f_test[[gene, 'Class']]
    grps = pd.unique(df_anova.Class.values)
    grps
    d_data = {grp: df_anova[gene][df_anova.Class == grp] for grp in grps}
    F, p = stats.f_oneway(d_data['LUAD'], d_data['PRAD'], d_data['BRCA'], d_data['KIRC'], d_data['COAD'])
    print("p_values:-", p)
    if p < 0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")

    return

f_test(df_f_test,"gene_6")

f_test(df_f_test,"gene_20522")

f_test(df_f_test,"gene_5")

df_cat_data = df
df_cat_data['Class'] = df_cat_data['Class'].map({'PRAD': 1, 'LUAD': 2, 'BRCA': 3, 'KIRC': 4, 'COAD': 5})
df_cat_data = df_cat_data.drop(['Unnamed: 0'],axis=1)

df_tsne_data = df
non_numeric = ['Unnamed: 0','Class']
df_tsne_data = df_tsne_data.drop(non_numeric, axis=1)
df_tsne_data