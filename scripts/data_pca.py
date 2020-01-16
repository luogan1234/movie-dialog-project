import numpy as np
from sklearn.decomposition import PCA

movie_emb = np.load('movie_emb.npz', allow_pickle=True)
embs = movie_emb['emb']
pca = PCA(n_components=768)
embs = pca.fit_transform(embs)
np.savez('movie_emb_pca.npz', emb=embs, dict=movie_emb['dict'])

