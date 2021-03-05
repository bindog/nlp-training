from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter


tencent_ai_word_embeddings = "/mnt/dl/public/word_embedding/Tencent_AILab_ChineseEmbedding.txt"
output_dir = "/mnt/dl/public/lmdb_embeddings/tencent_ai"

print("Loading word2vec from text file...")
w2v = KeyedVectors.load_word2vec_format(tencent_ai_word_embeddings, binary=False)

def iter_embeddings():
    for word in w2v.vocab.keys():
        yield word, w2v[word]

print('Writing vectors to a LMDB database...')
writer = LmdbEmbeddingsWriter(iter_embeddings()).write(output_dir)
