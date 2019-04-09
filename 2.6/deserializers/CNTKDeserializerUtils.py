import gzip
import re
from sre_parse import Pattern

import cntk
import numpy as np


def to_cntk_stream_infos(stream_indices):
    """
    Converts the stream indices list into a list of CNTK stream info objects.

    :param stream_indices:  The selected streams to emit, and the attribute indices they
                            correspond to.
    :return:                The StreamInformation objects as required by stream_infos(self).
    """

    # Initialise the stream-info list
    stream_infos = []

    # Process each set of stream indices in turn
    for i, stream_indices_pair in enumerate(stream_indices):
        name = stream_indices_pair[0]
        indices = stream_indices_pair[1]
        stream_infos.append(cntk.io.StreamInformation(name,
                                                      i,
                                                      'dense',
                                                      np.float32,
                                                      (len(indices),)))

    # Return the stream infos
    return stream_infos


def to_cntk_chunk(data, stream_indices):
    """
    Converts the raw data into CNTK format.

    :param data:            The raw data.
    :param stream_indices:  The columns to include with each stream.
    :return:                The streams of data as required by CNTK.
    """

    # Initialise the chunk
    chunk = {}

    # Add each stream in turn
    for name, indices in stream_indices:
        stream_data = [extract_by_index(row, indices) for row in data]
        stream_array = np.ascontiguousarray(stream_data, np.float32)
        chunk[name] = stream_array

    # Return the chunk
    return chunk


def get_open_func(filename):
    """
    Gets the open function to use to open the given filename.

    :param filename:    The file to open.
    :return:            The open function, and the read-mode flag to use.
    """

    # Select the open function based on the filename
    if filename.endswith('.gz'):
        return gzip.open, 'rt'
    else:
        return open, 'r'


def select_column_indices(column_names, num_columns, selection):
    """
    Returns a list of the indices of all selected columns.

    :param column_names:    The names of the columns.
    :param num_columns:     The number of columns available for selection.
    :param selection:       Either:
                            - An integer specifying a single column index.
                            - A pair of integers specifying an inclusive range of column indices.
                            - A string or Pattern specifying a regex to match against column names.
    :return:                The list of column indices selected.
    """

    # Process integer selections by normalisation
    if isinstance(selection, int):
        # Return the normalised selection as a list of itself
        return [normalise_array_index(selection, num_columns)]

    # Process tuple selections as a range of values
    elif isinstance(selection, tuple):
        # Get the normalised start and end indices
        start = normalise_array_index(selection[0], num_columns)
        end = normalise_array_index(selection[1], num_columns)

        # If end is before start, return the indices in reverse order
        step = -1 if end < start else 1

        # Return the inclusive range between start and stop, as a list
        return [i for i in range(start, end + step, step)]

    # Process strings and patterns by regex matching column names
    elif isinstance(selection, str) or isinstance(selection, Pattern):
        # Make sure we have column names to search through
        if column_names is None:
            raise ValueError('Cannot select columns by name if no names provided.')

        # Return any columns that match by name
        return [i
                for i, column_name in enumerate(column_names)
                if re.search(selection, column_name)]


def extract_by_index(a_list, indices):
    """
    Creates a list that consists of the elements at the given indices of
    the given list.

    :param a_list:      The list to get elements from.
    :param indices:     The indices to use to source elements.
    :return:            The selected list.
    """

    return [a_list[i] for i in indices]


def invert_indices(indices, size):
    """
    Returns the list of indices that are not in the given list, up to the given size.

    :param indices:     The list of indices to invert.
    :param size:        The size of the indexed list.
    :return:            A list of indices.
    """

    index_set = set(indices)
    return [i for i in range(size) if i not in index_set]


def normalise_array_index(index, length):
    """
    Normalises the given list index into the valid range for a
    list of the given length. This way negative indices index
    backward from the end of the list.

    :param index:   The index to normalise.
    :param length:  The list length to normalise to.
    :return:        The normalised index.
    """
    return index % length
