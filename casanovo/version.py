"""Get the package version"""


def _get_version():
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]

    """
    try:
        # Fast, but only works in Python 3.8+
        from importlib.metadata import version, PackageNotFoundError

        try:
            return version("casanovo")
        except PackageNotFoundError:
            return None

    except ImportError:
        # Slow, but works for all Python 3+
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            return get_distribution("casanovo").version
        except DistributionNotFound:
            return None
