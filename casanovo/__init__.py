import logging

from .version import _get_version


__version__ = _get_version()

# Filter tenserflow warnings on google Collab
try:
    import tensorflow as tf

    tf.get_logger().setLevel(logging.CRITICAL)
    print("FILTER")
except:
    print("NO FILTER")
    pass
