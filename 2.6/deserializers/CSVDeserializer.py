import csv

import numpy as np
from cntk.io import UserDeserializer

import CNTKDeserializerUtils
import onehot

# Number of lines to sample to try and determine CSV format
SAMPLE_SIZE = 4

# The data-type to use for numeric values
NUMERIC_TYPE = np.float32


class CSVDeserializer(UserDeserializer):
    def __init__(self, filename, streams,
                 encoding='utf-8',
                 has_header=None,
                 numeric_columns=None,
                 **fmtparams):
        super(CSVDeserializer, self).__init__()

        # Select the open method based on the filename
        open_func, mode = CNTKDeserializerUtils.get_open_func(filename)

        # Read in the file
        with open_func(filename, mode, encoding=encoding, newline='') as file:
            # Sniff the possible dialect
            dialect, has_header = sniff_dialect(file, has_header)

            # Get the reader
            reader = csv.reader(file, dialect, **fmtparams)

            # Read in all the data
            data = [row for row in reader]

        # Take the header from the data if there is one
        if has_header:
            self.__header = data.pop(0)
        else:
            self.__header = None

        # Select the columns for each stream
        streams = [(name, CNTKDeserializerUtils.select_column_indices(self.__header, len(data[0]), selection))
                   for name, selection in streams]

        # Convert numeric columns into numeric format
        if numeric_columns is None:
            numeric_columns = estimate_numeric_columns(data[:SAMPLE_SIZE])
        convert_numeric_columns(data, numeric_columns)

        # Create the one-hot mappings for each string column
        string_columns = CNTKDeserializerUtils.invert_indices(numeric_columns, len(data[0]))
        mapping = onehot.Mapping(len(data[0]))
        for column_index in string_columns:
            mapping.add_encoding(column_index, onehot.Encoding(get_column(data, column_index), NUMERIC_TYPE))

        # Apply the one-hot encoding to the header
        if self.__header is not None:
            self.__header = mapping.encode_header(self.__header)

        # One-hot encode the data
        mapping.encode(data)

        # One hot encode the stream selections
        streams = [(name, mapping.encode_column_selection(selection))
                   for name, selection in streams]

        # Create the CNTK stream infos
        self.__stream_infos = CNTKDeserializerUtils.to_cntk_stream_infos(streams)

        # Create the data chunk
        self.__chunk = CNTKDeserializerUtils.to_cntk_chunk(data, streams)

    def stream_infos(self):
        # Return the pre-processed stream infos
        return self.__stream_infos

    def num_chunks(self):
        # Entire file is loaded to memory, so only ever 1 chunk
        return 1

    def get_chunk(self, chunk_id):
        # Only one chunk, so only chunk id 0 is valid
        if chunk_id != 0:
            raise IndexError('chunk_id must be 0 (' + str(chunk_id) + ' provided)')

        # Return the one chunk
        return self.__chunk


def sniff_dialect(file, has_header):
    """
    Uses the csv Sniffer class to try and auto-detect the dialect of
    the CSV file. Also determines if the file has a header row.

    :param file:        The CSV file to sniff.
    :param has_header:  Whether the file has a header, or None to auto-detect.
    :return:            The detected dialect of the file, and whether it has a header.
    """

    # Peek a small sample of lines from the file
    sample = ''.join([file.readline() for _ in range(SAMPLE_SIZE)])
    file.seek(0)

    # Create the sniffer
    sniffer = csv.Sniffer()

    # Sniff the dialect
    dialect = sniffer.sniff(sample)

    # Sniff the header if it isn't already determined
    if has_header is None:
        has_header = sniffer.has_header(sample)

    return dialect, has_header


def estimate_numeric_columns(sample):
    """
    Attempts to guess which columns of the provided sample contain numeric data.

    :param sample:  A sample of the data from the CSV file.
    :return:        A list of indices of columns which are thought to contain numeric data.
    """

    # Initialise the list
    numeric_columns = []

    # Test each column in turn
    for column_index in range(len(sample[0])):
        try:
            # Attempt to convert all values in the column into numerics
            for row_index in range(len(sample)):
                value = NUMERIC_TYPE(sample[column_index][row_index])
        except:
            # If any fails, assume it's not a numeric column
            continue

        # If all values converted, assume it is a numeric column
        numeric_columns.append(column_index)

    return numeric_columns


def convert_numeric_columns(data, numeric_columns):
    """
    Converts the values in each of the specified columns into numerics.

    :param data:                The data from the CSV file.
    :param numeric_columns:     The list of indices of numeric columns.
    :return:                    Nothing, the data is converted in place.
    """

    # Convert each row in turn
    for row in data:
        # Convert only those columns that are marked numeric
        for column_index in numeric_columns:
            # Convert the value in place
            row[column_index] = NUMERIC_TYPE(row[column_index])


def get_column(data, column_index):
    """
    Gets a column of data from the given data.

    :param data:            The data from the CSV file.
    :param column_index:    The column to copy.
    :return:                The column of data (as a list).
    """
    return [row[column_index] for row in data]
