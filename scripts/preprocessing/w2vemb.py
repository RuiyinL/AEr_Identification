import random
from w2vembeddings.embedding import Embedding
from os import path


class EMB(Embedding):
    def __init__(self, name, dimensions, default='none'):
        assert default in {'none', 'random', 'zero'}
        self.default = default
        self.dimensions = dimensions

        self.db = self.initialize_db(self.path(path.join(name, '{}:{}.db'.format(name, dimensions))))

    # def embedding_dic_check(self, word):
    #     vector = self.lookup(word)
    #     if vector is not None:
    #         return True
    #     else:
    #         return False

    def get_vector(self, word, default=None):
        if default is None:
            default = self.default
        get_default = {
            'none': lambda: None,
            'zero': lambda: 0.,
            'random': lambda: random.uniform(-0.1, 0.1),
        }[default]
        g = self.lookup(word)
        return [get_default() for i in range(self.dimensions)] if g is None else g
