# coding: utf-8
"""
Every NLP task needs a Vocabulary
Every Vocabulary is built from Instances
Every Instance is a collection of Fields
"""

__all__ = ['DefaultLookupDict', 'Vocabulary']

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_IDX = 0
UNK_IDX = 1


class DefaultLookupDict(dict):
    def __init__(self, default):
        super(DefaultLookupDict, self).__init__()
        self._default = default

    def __getitem__(self, item):
        return self.get(item, self._default)


class Vocabulary:
    """
    Define a vocabulary object that will be used to numericalize a field.
    Attributes:
        token2id: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        id2token: A list of token strings indexed by their numerical
        identifiers.
        embedding: pretrained vectors.

    Examples:
    >>> from torchlight.vocab import Vocabulary
    >>> from collections import Counter
    >>> text_data = ['hello', 'world', 'hello', 'nice', 'world', 'hi', 'world']
    >>> vocab = Vocabulary(Counter(text_data))
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=None):
        """
        Create a Vocabulary given Counter.
        Args:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens except ['<pad>', '<unk>'].
                Possible choices: [CLS] [MASK] [SEP] in BERT or <bos> <eos>
                in Machine Translation.
        """
        min_freq = max(min_freq, 1)  # must be positive

        if specials is None:
            self.specials = [PAD_TOKEN, UNK_TOKEN]
        else:
            assert isinstance(specials, list), "'specials' is of type list"
            self.specials = [PAD_TOKEN, UNK_TOKEN] + specials

        assert len(set(self.specials)) == len(self.specials), \
            "specials can not contain duplicates."

        if max_size is not None:
            max_size = len(self.specials) + max_size

        self.id2token = self.specials[:]
        self.token2id = DefaultLookupDict(UNK_IDX)
        self.token2id.update({tok: i for i, tok in enumerate(self.id2token)})

        # sort by frequency, then alphabetically
        token_freqs = sorted(counter.items(), key=lambda tup: tup[0])
        token_freqs.sort(key=lambda tup: tup[1], reverse=True)

        for token, freq in token_freqs:
            if freq < min_freq or len(self.id2token) == max_size:
                break
            if token not in self.specials:
                self.id2token.append(token)
                self.token2id[token] = len(self.id2token) - 1

        # TODO
        self.embedding = None

    def __len__(self):
        return len(self.id2token)

    def __repr__(self):
        return 'Vocab(size={}, specials="{}")'.format(len(self), self.specials)

    def __getitem__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.
        If `unknown_token` of the vocabulary is None, looking up unknown tokens
        results in KeyError.
        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.
        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        if not isinstance(tokens, (list, tuple)):
            return self.token2id[tokens]
        else:
            return [self.token2id[token] for token in tokens]

    def __call__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.
        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.
        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the
            vocabulary.
        """

        return self[tokens]

    @classmethod
    def from_json(cls, json_str):
        pass

    def to_json(self):
        pass

    def set_embedding(self):
        pass
