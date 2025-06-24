"""
Utilities
=========

Common helper functions used throughout the Axon SDK.
"""

def flatten_nested_list(nested_list: list) -> list:
    """
        Recursively flattens an arbitrarily nested list into a single flat list.

        This function supports arbitrary nesting of Python lists and returns a
        single list containing all the elements in depth-first order.

        Args:
            nested_list (list): A list that may contain other lists as elements.

        Returns:
            list: A flat list with all nested elements extracted.

        Example:
            >>> flatten_nested_list([1, [2, [3, 4], 5]])
            [1, 2, 3, 4, 5]
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list
