"""
Data Encoding
=============

This module defines the `DataEncoder` class for converting continuous values into
spike intervals and decoding them back, based on the timing interval coding scheme
used in the STICK model.

Classes:
    - DataEncoder: Encodes/decodes values as timing intervals between two spikes.
"""


class DataEncoder:
    """
    Encodes values into inter-spike intervals and decodes them back.

    The STICK model uses spike timing to represent scalar values in a time-based manner.
    This class provides conversion from values ∈ [0, 1] to spike timings and vice versa.

    Attributes:
        Tmin (float): Minimum inter-spike interval, corresponding to value 0.
        Tcod (float): Encoding range for values in milliseconds.
        Tmax (float): Maximum spike interval (Tmin + Tcod), corresponding to value 1.
    """
    def __init__(self, Tmin=10.0, Tcod=100.0):
        """
        Initialize the encoder with the given minimum interval and coding range.

        Args:
            Tmin (float): Minimum spike interval in milliseconds (default is 10.0).
            Tcod (float): Maximum coding range above Tmin in milliseconds (default is 100.0).
        """
        self.Tmin = Tmin
        self.Tcod = Tcod
        self.Tmax = Tmin + Tcod

    def encode_value(self, value: float) -> tuple[float, float]:
        """
        Encode a value into a pair of spike times.

        The spike interval encodes the magnitude of the value. The first spike is always at t=0.

        Args:
            value (float): A normalized value in the range [0, 1].

        Returns:
            tuple[float, float]: A pair of spike times (t0, t1) such that (t1 - t0) encodes the value.

        Raises:
            AssertionError: If the value is not in [0, 1].
        """
        assert value >= 0 and value <= 1
        interval = self.Tmin + value * self.Tcod
        return (0, interval)

    def decode_interval(self, spiking_interval: float) -> float:
        """
        Decode an inter-spike interval back into a normalized value.

        Args:
            spiking_interval (float): The time between two spikes.

        Returns:
            float: A value in [0, 1] corresponding to the spike interval.
        """
        value = (spiking_interval - self.Tmin) / self.Tcod
        return value
