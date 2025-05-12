import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Afficher 5 formes aléatoires de chaque classe
def plotRandomForms(df, x, form, n=20):
	# Il y a n courbes de chaque forme dans le DataFrame
	dict = {"M": 0, "Spike": 1, "CLDR": 2, "CRDL": 3, "Parabola": 4}
	j = dict[form]
	for i in range(5):
		idx = random.randint(j*n, n*j+19)
		plt.plot(x, df['Form'][idx])
	plt.title(f"5 random {form}")
	# plt.show()
    
# Afficher 5 formes par classe
def plotAllForms(df, x, n=20):
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 2, 1)
    plotRandomForms(df, x, "M", n)
    plt.subplot(3, 2, 2)
    plotRandomForms(df, x, "Spike", n)
    plt.subplot(3, 2, 3)
    plotRandomForms(df, x, "CLDR", n)
    plt.subplot(3, 2, 4)
    plotRandomForms(df, x, "CRDL", n)
    plt.subplot(3, 2, 5)
    plotRandomForms(df, x, "Parabola", n)
    plt.show()

# Afficher la matrice de confusion
def plot_confusion_matrix(df, confusion_matrix):
	"""
	# Créer une figure
	# plt.figure(figsize=(8, 8))

	# Afficher la matrice de confusion
	plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))

	# Ajouter une barre de couleur
	# plt.colorbar()

	# Ajouter les labels des axes
	classes = ['M', 'Spike', 'CLDR', 'CRDL', 'Parabola']
	plt.xticks(np.arange(len(classes)), classes, rotation=45)
	plt.yticks(np.arange(len(classes)), classes)

	# Afficher les valeurs dans la matrice
	for i in range(confusion_matrix.shape[0]):
		for j in range(confusion_matrix.shape[1]):
			plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='white')

	"""
	ConfusionMatrixDisplay(confusion_matrix, display_labels=df["Predicted Class"].unique()).plot(colorbar=True)
	# Ajouter les labels
	plt.xlabel('Classes prédites')
	plt.ylabel('Classes réelles')
	# Ajouter l'accuracy dans le titre
	plt.title(f'Matrice de confusion (Accuracy: {np.trace(confusion_matrix) / np.sum(confusion_matrix):.2f})')
	plt.show()

# Afficher quelques résultats de clustering
def plot_clusters(df, x, k, n=10, ukmeans = False):
	if ukmeans :
		title = f'Clustering avec UK-means (clusters={k})'
	else:
		title = f'Clustering avec K-means (k={k})'
	# Select n random indices from the DataFrame
	random_indices = np.random.choice(df.index, size=n, replace=False)
	selected_forms = df.loc[random_indices]

	# Define colors for each cluster
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	color_map = {i: colors[i % len(colors)] for i in range(k)}

	# Plot each selected form on the same graph
	plt.figure(figsize=(10, 6))
	for index, row in selected_forms.iterrows():
		plt.plot(x, row['Form'], color=color_map[row['Predicted Cluster']], label=f'Form {index} - Cluster {row["Predicted Cluster"]}')

	plt.title(title)
	plt.xlabel('x')
	plt.ylabel('Amplitude')
	plt.legend()
	plt.show()


# Afficher les courbes mal classées
def plot_misclassified(df, x):
	misclassified = df[df['Class'] != df['Predicted Class']]
	for index, row in misclassified.iterrows():
		plt.plot(x, row['Form'], label=f"Form {index} - Actual: {row['Class']} - Predicted: {row['Predicted Class']}")
	plt.title('Misclassified Forms')
	# plt.legend()
	plt.show()


def plot_external_metrics(metrics):
	plt.figure(figsize=(8, 6))
	for metric in ['accuracy', 'ari', 'nmi', 'fmi', 'jaccard']:
		plt.plot(metrics['k_range'], metrics[metric], marker='o', label=metric.capitalize())
	plt.legend()
	plt.title('External Evaluation Metrics')
	plt.xlabel('Number of Clusters')
	plt.ylabel('Score')
	plt.show()

def plot_internal_metrics(metrics):
	fig, ax1 = plt.subplots(figsize=(9, 6))
	k_range = metrics["k_range"]

	# Dessiner la métrique Silhouette et Davies-Bouldin
	ax1.plot(k_range, metrics["silhouette"], marker='o', color='tab:blue',label='Silhouette')
	ax1.plot(k_range, metrics["davies_bouldin"], 'b-o', label='Davies-Bouldin')
	ax1.set_xlabel('Number of Clusters')
	ax1.set_ylabel('Silhouette Score', color='tab:blue')
	ax1.tick_params(axis='y', labelcolor='tab:blue')

	# Créer un second axe pour la métrique Calinski-Harabasz
	ax2 = ax1.twinx()  
	ax2.set_ylabel('Calinski-Harabasz Score', color='tab:red')  
	ax2.plot(k_range, metrics["calinski_harabasz"], marker='o', color='tab:red', label='Calinski-Harabasz')
	ax2.tick_params(axis='y', labelcolor='tab:red')

	fig.tight_layout(pad=3)

	# Ajouter une légende
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc='upper left')
	plt.title('Internal Evaluation Metrics')
	plt.show()

def plot_external_and_sse(metrics):
	fig, ax1 = plt.subplots(figsize=(9, 6))
	fig.subplots_adjust(top=0.9)
	k_range = metrics["k_range"]
	# Dessiner les métriques externes
	ax1.plot(k_range, metrics["accuracy"], marker='o', color='tab:blue',label='Accuracy')
	ax1.plot(k_range, metrics["ari"], 'g-o', label='ARI')
	ax1.plot(k_range, metrics["nmi"], 'c-o', label='NMI')
	ax1.plot(k_range, metrics["fmi"], 'm-o', label='FMI')
	ax1.plot(k_range, metrics["jaccard"], 'k-o', label='Jaccard')
	ax1.set_xlabel('Number of Clusters')
	ax1.set_ylabel('Score', color='tab:blue')
	ax1.tick_params(axis='y', labelcolor='tab:blue')

	# Créer un second axe pour la métrique SSE
	ax2 = ax1.twinx()
	ax2.set_ylabel('SSE', color='tab:red')
	ax2.plot(k_range, metrics["sse"], marker='o', color='tab:red', label='SSE')
	ax2.tick_params(axis='y', labelcolor='tab:red')

	fig.tight_layout(pad=3)
	# Ajouter une légende
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc='upper left')
	plt.title('External Evaluation Metrics and SSE')
	plt.show()