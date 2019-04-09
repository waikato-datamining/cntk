from cntk.io import UserDeserializer

import onehot
from ARFFFile import NOMINAL_KEYWORD, NUMERIC_KEYWORD, NUMERIC_TYPE
import CNTKDeserializerUtils


class ARFFDeserializer(UserDeserializer):
    """
    CNTK custom deserializer for reading in ARFF files. Does not support
    relation attributes.
    """

    def __init__(self, arff_file, streams):
        """
        Initialises the ARFFDeserialiser.

        :param arff_file:       The ARFF file to pull data from.
                                N.B. The ARFFFile object is modified by this method.
        :param streams:         A list of streams to emit. Each entry should be a pair of
                                the form (name, regex) or (name, (first, last)).
        """
        # Initialise the super-class
        super(ARFFDeserializer, self).__init__()

        # Create the stream map by index
        stream_indices = create_stream_index_map(arff_file, streams)

        # Make sure selected attribute streams only have numeric/nominal attributes
        check_attribute_types(stream_indices, arff_file.attributes)

        # One-hot encode the file
        one_hot_map = one_hot_encode(arff_file)

        # Apply the one-hot encoding to the stream attribute selection
        stream_indices = [(name, one_hot_map.encode_column_selection(indices))
                          for name, indices in stream_indices]

        # Remember how many attributes are in each stream
        self.__stream_sizes = create_stream_size_map(stream_indices)

        # Parse the header for the stream infos
        self.__stream_infos = CNTKDeserializerUtils.to_cntk_stream_infos(stream_indices)

        # Parse the data into CNTK format
        self.__chunk = CNTKDeserializerUtils.to_cntk_chunk(arff_file.data, stream_indices)

    def get_number_of_attributes(self, stream):
        """
        Gets the number of attributes for a given stream name.

        :param stream:  The name of the stream to check.
        :return:        The number of attributes in the given stream.
        """

        if stream in self.__stream_sizes:
            return self.__stream_sizes[stream]
        else:
            return 0

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


def create_stream_size_map(stream_indices):
    """
    Creates a map from stream name to stream size from the indices table.

    :param stream_indices:  The stream indices table.
    :return:                The map.
    """
    return {pair[0]: len(pair[1]) for pair in stream_indices}


def check_attribute_types(stream_indices, attributes):
    """
    Checks that all attributes in the selected streams are either numeric or nominal type.

    :param stream_indices:      The selected stream indices.
    :param attributes:          The set of attributes to check.
    :return:                    Nothing, but raises an error if any attribute is not numeric or nominal.
    """

    # Create the set of all selected indices
    indices_set = set()
    for pair in stream_indices:
        for index in pair[1]:
            indices_set.add(index)

    # Check each selected attribute
    for index in indices_set:
        # Get the attribute's type
        attribute_type = attributes[index]['type']

        # If it's not numeric or nominal, raise an exception
        if attribute_type != NOMINAL_KEYWORD and attribute_type != NUMERIC_KEYWORD:
            raise UnsupportedAttributeTypeError(attribute_type)


def one_hot_encode(arff_file):
    """
    Modifies the ARFF file to encode its nominal attributes as numeric attributes
    using one-hot encoding.

    :param arff_file:   The ARFFFile object to modify.
    :return:            The number of class attributes after one-hot encoding.
    """

    # Create the mapping from nominal to numeric attributes
    one_hot_map = create_one_hot_map(arff_file.attributes)

    # Apply the mapping to the attributes
    arff_file.attributes = one_hot_encode_attributes(arff_file.attributes, one_hot_map)

    # Apply the mapping to the data
    one_hot_map.encode(arff_file.data)

    # Return the one-hot encoding
    return one_hot_map


def create_one_hot_map(attributes):
    """
    Creates a list of one-hot encoding maps for the given attributes.

    :param attributes:  The attributes being encoded.
    :return:            The list of one-hot mappings.
    """

    # Create the map
    one_hot_map = onehot.Mapping(len(attributes))

    # Process each attribute
    for attribute_index, attribute in enumerate(attributes):
        # If the attribute is nominal, add a mapping from value name to encoding
        if attribute['type'] == NOMINAL_KEYWORD:
            values = attribute['values']

            # Append the mapping structure to the one-hot map
            one_hot_map.add_encoding(attribute_index, onehot.Encoding(values, NUMERIC_TYPE))

    # Return the map
    return one_hot_map


def one_hot_encode_attributes(attributes, one_hot_map):
    """
    Uses the one-hot encoding map to modify the given attribute list into its
    one-hot encoded form.

    :param attributes:      The attributes to encode.
    :param one_hot_map:     The one-hot encoding map to use for the encoding.
    :return:                The encoded list of attributes.
    """

    attribute_names = [attribute['name'] for attribute in attributes]
    attribute_lookup = {attribute['name']: attribute for attribute in attributes}

    new_names = one_hot_map.encode_header(attribute_names)

    for name in new_names:
        if name not in attribute_lookup:
            attribute_lookup[name] = {'name': name, 'type': NUMERIC_KEYWORD}

    return [attribute_lookup[name] for name in new_names]


def create_stream_index_map(arff_file, streams):
    """
    Converts a list of stream selections to a list of stream name => attribute indices
    for the given ARFF file.

    :param arff_file:       The ARFFFile object to select indices from.
    :param streams:         The map of stream name => regex.
    :return:                The map of stream name => indices.
    """
    return [(name, arff_file.select_attribute_indices(selection)) for name, selection in streams]


class ARFFDeserializerError(Exception):
    """
    Base class for all exceptions arising from using the ARFFDeserializer class.
    """

    def __init__(self, message):
        super(ARFFDeserializerError, self).__init__(message)


class UnsupportedAttributeTypeError(ARFFDeserializerError):
    """
    Exception for attribute types that the deserializer doesn't support.
    """

    def __init__(self, attribute_type):
        message = 'ARFFDeserializer supports only numeric/nominal attributes, not ' + attribute_type
        super(UnsupportedAttributeTypeError, self).__init__(message)
