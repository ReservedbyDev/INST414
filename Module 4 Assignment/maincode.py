import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
import spotipy
from spotipy.oauth2 import SpotifyOAuth

os.environ['SPOTIPY_CLIENT_ID'] = 'f55c0d3af99741d3ae48b56471e4c06f'
os.environ['SPOTIPY_CLIENT_SECRET'] = '71b7919bd7b9400ea03ef9252b23b3e6'
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://127.0.0.1:8888/callback'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope='user-top-read'))

def get_user_genre(user_label='user_1', top_n_genres=10):
    results = sp.current_user_top_tracks(limit=50, time_range='medium_term')
    genre_counts = Counter()
    genre_cache = {}

    for item in results['items']:
        for artist in item['artists']:
            artist_id = artist['id']
            if artist_id in genre_cache:
                genres = genre_cache[artist_id]
            else:
                try:
                    artist_data = sp.artist(artist_id)
                    genres = artist_data['genres']
                    genre_cache[artist_id] = genres
                except:
                    continue
            genre_counts.update(genres)

    if not genre_counts:
        return None

    top_genres = dict(genre_counts.most_common(top_n_genres))
    return pd.Series(top_genres, name=user_label)

base_user = get_user_genre('user_1', top_n_genres=10)
if base_user is None:
    exit()

genre_df = pd.DataFrame()
for i in range(5):
    randomized_user = base_user.copy()
    randomized_user += np.random.randint(-2, 3, size=base_user.shape)
    randomized_user = randomized_user.clip(lower=0)
    randomized_user.name = f'user_{i+1}'
    genre_df = pd.concat([genre_df, randomized_user], axis=1)

genre_df = genre_df.fillna(0).T
normalized = normalize(genre_df)
distance_matrix = pairwise_distances(normalized, metric='cosine')

silhouette_scores = []
k_range = range(2, 5)
for k in k_range:
    model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    labels = model.fit_predict(distance_matrix)
    score = silhouette_score(distance_matrix, labels, metric='precomputed')
    silhouette_scores.append(score)

plt.figure()
plt.plot(k_range, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Score vs. Number of Clusters (k)')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('spotify silhouette.png')
plt.show()

optimal_k = k_range[np.argmax(silhouette_scores)]
final_model = AgglomerativeClustering(n_clusters=optimal_k, metric='precomputed', linkage='average')
genre_df['cluster'] = final_model.fit_predict(distance_matrix)

pca = PCA(n_components=2)
components = pca.fit_transform(normalized)
genre_df['pca1'] = components[:, 0]
genre_df['pca2'] = components[:, 1]

plt.figure()
sns.scatterplot(data=genre_df, x='pca1', y='pca2', hue='cluster')
plt.title('Spotify Users Clustered by Genre Preference (PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('real spotify pca.png')
plt.show()

genre_df.to_csv('spotify user clusters.csv')
