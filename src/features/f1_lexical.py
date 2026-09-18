"""F1: TF-IDF token + n-gram → SVD 256."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from config import F1_TFIDF_MAX_FEATURES, F1_NGRAM_RANGE, FUSION_DIM, DATA_DIR
from src.utils import ensure_dir

def extract_f1(tr_codes, va_codes, te_codes, name):
    vec = TfidfVectorizer(max_features=F1_TFIDF_MAX_FEATURES,
                          ngram_range=F1_NGRAM_RANGE,
                          token_pattern=r'\S+', lowercase=False)
    X_tr = vec.fit_transform(tr_codes)
    svd = TruncatedSVD(n_components=FUSION_DIM, random_state=42)
    Z_tr = svd.fit_transform(X_tr).astype(np.float32)
    Z_va = svd.transform(vec.transform(va_codes)).astype(np.float32)
    Z_te = svd.transform(vec.transform(te_codes)).astype(np.float32)
    out = ensure_dir(DATA_DIR / name / 'features' / 'F1')
    np.save(out / f'{name}_train_f1.npy', Z_tr)
    np.save(out / f'{name}_val_f1.npy',   Z_va)
    np.save(out / f'{name}_test_f1.npy',  Z_te)
    return Z_tr, Z_va, Z_te