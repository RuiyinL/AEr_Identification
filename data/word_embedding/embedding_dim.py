import fasttext.util
fasttext.FastText.eprint = lambda x: None
# from gensim.models.keyedvectors import KeyedVectors

# load pre-trained word embedding
ft = fasttext.load_model('D:\scripts\word_embedding\cc.en.300.bin')
fasttext.util.reduce_model(ft, 100)     # 100 dimension
# fasttext.util.reduce_model(ft, 200)     # 200 dimension

ft.save_model('D:\scripts\word_embedding\cc.en.100.bin')
# ft.save_model('D:\scripts\word_embedding\cc.en.200.bin')