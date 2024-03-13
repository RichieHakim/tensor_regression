__all__ = [
    'convolutional_reduced_rank_regression',
]

for pkg in __all__:
    exec('from . import ' + pkg)

__version__ = '0.2.1'
