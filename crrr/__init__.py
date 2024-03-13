from crrr.model import Convolutional_Reduced_Rank_Regression

__all__ = [
    'model',
]

for pkg in __all__:
    exec('from . import ' + pkg)

__version__ = '0.2.1'
