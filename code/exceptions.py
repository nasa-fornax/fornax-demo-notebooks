class CutoutError(Exception):
    """Raised when the cutout cannot be extracted."""

    pass


class TractorError(Exception):
    """Raised when the Tractor optimization fails to converge."""

    pass
