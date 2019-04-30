# Keywords for the different sections of the ARFF file
import re
from multiprocessing.pool import Pool

import numpy as np

# The data-type to use for numeric values
import CNTKDeserializerUtils

# Numeric type to use for numeric data
import onehot

NUMERIC_TYPE = np.float32

# The number of rows to process as a batch in parallel mode
PARALLEL_BATCH_SIZE = 64

# The keywords for the different sections of the file
RELATION_KEYWORD = '@relation'
ATTRIBUTE_KEYWORD = '@attribute'
DATA_KEYWORD = '@data'

# Keywords for the different attribute datatypes
# (relational attributes are not supported)
NUMERIC_KEYWORD = 'numeric'
INTEGER_KEYWORD = 'integer'
REAL_KEYWORD = 'real'
NOMINAL_KEYWORD = 'nominal'
DATE_KEYWORD = 'date'
STRING_KEYWORD = 'string'

# The symbol that represents a missing value
MISSING_VALUE_SYMBOL = '?'

# The symbol that represents the beginning of a comment
COMMENT_SYMBOL = '%'

# The set of characters that can delimit a data value
DATA_DELIMITERS = {'\t', ',', '\n'}

# The set of quotation symbols
QUOTES = {'\"', '\''}

# Regular expression for matching a name (for a relation or attribute)
# (possibly surrounded by quotes)
NAME_PATTERN = '''
    (?P<quote>['"])?            # An optional beginning quote
    (?![{},%])                  # Must not start with '{', '}', ',' or '%'
    (?(quote)                   # If we are in quotes...
        (\\w|\\s|-|\\.)+        # Match any words, spaces, dashes or dots
        |                       # If we are not in quotes...
        (\\w|-|\\.)+            # The same, but don't match spaces
    )
    (?(quote)(?P=quote))        # Match the same ending quote as was found at the beginning (if there was one)
    '''

# Regular expression for matching the name in a line
# (possibly surrounded by quotes)
NOMINAL_VALUE_PATTERN = '''
    (?P<quote>['"])?            # An optional beginning quote
    (?(quote)                   # If we are in quotes...
        .+?                     # Match any characters up to the close quotation
        |                       # If we are not in quotes...
        [^\\s,}]+               # Match to the next space or delimiter
    )
    (?(quote)(?P=quote))        # Match the same ending quote as was found at the beginning (if there was one)
    '''

# Matches the attribute type portion of an attribute declaration
ATTRIBUTE_TYPE_PATTERN = '''
    (numeric|integer|real|string|date|{.*?})    # A type keyword or a nominal specification
    '''

# Regular expression which matches a data row value and its delimiter
DATA_VALUE_PATTERN = '''
    (?P<quote>['"])?            # An optional beginning quote
    .*?                         # Any characters can exist in the value (will be checked separately)
    (?(quote)(?P=quote))        # Match the same ending quote as was found at the beginning (if there was one)
    \\s*                        # Some possible trailing whitespace
    (\\t|,|\\n)                 # The delimiter is a tab, a comma, or a new line
    '''

# Regular expression which matches a date-format string
# TODO: Implement properly. csterling
DATE_FORMAT_PATTERN = '''
    .*          # Just match anything for now
    '''


class ARFFFile:
    """
    Represents an ARFF file in memory. Loads the file into memory on initialisation
    so it can be closed straight away.

    Currently does not handle relational attributes or date formats.
    """

    def __init__(self, filename, encoding='utf-8', parallel=False):
        # Select the open method based on the filename
        open_func, mode = CNTKDeserializerUtils.get_open_func(filename)

        # Parse the file
        with open_func(filename, mode, encoding=encoding) as file:
            # Get the relation and attributes
            self.relation = get_relation_section(file)
            self.attributes = get_attribute_section(file)

            # Select the data-reading method (serial or parallel)
            data_section_method = get_data_section_parallel if parallel else get_data_section

            # Get the data
            self.data = data_section_method(file, self.attributes)

        # Create the attribute name lookup
        self.__attribute_name_lookup = {attribute['name']: i
                                        for i, attribute in enumerate(self.attributes)}

    def remove_attribute(self, attribute):
        """
        Removes an attribute from the file and the associated data.

        :param attribute:   The index or name of the attribute to remove.
        :return:            Nothing.
        """

        # Normalise the name to an index
        attribute = self.normalise_attribute_reference(attribute)

        # Delete the attribute
        self.attributes.pop(attribute)

        # Delete the data
        for row in self.data:
            row.pop(attribute)

    def select_attribute_indices(self, selection):
        """
        Selects the indices of one or more attributes.

        :param selection:   The selection criteria.
        :return:            A list of attribute indices.
        """

        # Create a list of the attribute names
        attribute_names = [attribute['name'] for attribute in self.attributes]

        # Defer
        return CNTKDeserializerUtils.select_column_indices(attribute_names, len(attribute_names), selection)

    def row_count(self):
        """
        Gets the number of rows in the file.

        :return:    The number of rows.
        """
        return len(self.data)

    def attribute_count(self):
        """
        Gets the number of attributes in the file.

        :return:    The number of attributes.
        """
        return len(self.attributes)

    def map_string_attribute(self, attribute):
        """
        Converts a string attribute to a numerical attribute where each entry is
        the index of the original string in a lookup table.

        :param attribute:   The string attribute to map.
        :return:            The lookup table.
        """

        # Normalise the attribute reference
        attribute = self.normalise_attribute_reference(attribute)

        # Create the mapping
        index_map, string_table = self.create_string_index_map(attribute)

        # Make sure mapping was successful
        if string_table is None:
            return None

        # Change the attribute type
        self.attributes[attribute]['type'] = NUMERIC_KEYWORD

        # Apply the mapping to the data
        for row in self.data:
            row[attribute] = NUMERIC_TYPE(index_map[row[attribute]])

        # Return the lookup table
        return string_table

    def create_string_index_map(self, attribute):
        """
        Creates a mapping from string values to indices for a string attribute
        or a nominal attribute.

        :param attribute:   The attribute to map (index or name).
        :return:            The index map for the strings in the given attribute.
        """

        # Normalise the attribute reference
        attribute = self.normalise_attribute_reference(attribute)

        # Make sure the attribute is of type string
        attribute_type = self.attributes[attribute]['type']
        if attribute_type != STRING_KEYWORD and attribute_type != NOMINAL_KEYWORD:
            return None, None

        # Create the map
        index_map = {}
        string_table = []

        # Add each value to the map
        for row in self.data:
            string_value = row[attribute]
            if string_value not in index_map:
                index_map[string_value] = len(string_table)
                string_table.append(string_value)

        # Return the map
        return index_map, string_table

    def one_hot_encode(self):
        """
        Applies a one-hot encoding to all nominal attributes in the file.

        :return:    The mapping used to perform the encoding.
        """
        return one_hot_encode(self)

    def normalise_attribute_reference(self, attribute):
        """
        Takes an attribute name or index and returns the valid index of the attribute.

        :param attribute:   The name or index of the attribute to find.
        :return:            The index of the attribute if valid, otherwise None.
        """
        if isinstance(attribute, str) and attribute in self.__attribute_name_lookup:
            return self.__attribute_name_lookup[attribute]
        elif isinstance(attribute, int):
            return CNTKDeserializerUtils.normalise_array_index(attribute, len(self.attributes))
        else:
            return None


def get_relation_section(file):
    """
    Gets the relation name from the ARFF file.

    :param file:    The file we are processing.
    :return:        The relation name.
    """

    # Get the line starting with the @relation keyword
    line = read_till_found(file, {RELATION_KEYWORD})

    # Parse the line
    relation = parse_relation_line(line)

    # Return the header
    return relation


def get_attribute_section(file):
    """
    Gets the list of attributes in the ARFF file.

    :param file:    The ARFF file being processed.
    :return:        The list of attributes in the file.
    """

    # Create the empty list of attributes
    attributes = []

    # Keep reading attributes until we find the @data keyword
    line = read_till_found(file, {ATTRIBUTE_KEYWORD, DATA_KEYWORD})
    while not line_starts_with(line, DATA_KEYWORD):
        # Parse the @attribute line
        attribute = parse_attribute_line(line)

        # Add it to the list
        attributes.append(attribute)

        # Read the next attribute or data line
        line = read_till_found(file, {ATTRIBUTE_KEYWORD, DATA_KEYWORD})

    # Return the attributes
    return attributes


def get_data_section(file, attributes):
    """
    Gets the data from the data section of the ARFF file.

    :param file:        The ARFF file we are processing.
    :param attributes:  The attributes of the ARFF file.
    :return:            The list of data rows.
    """

    # Create the empty data list
    data = []

    # Keep reading lines until we run out of file
    line = None
    while line != '':
        # Read the next line
        line = file.readline()

        # If it was just whitespace/comments, skip it
        if is_whitespace_only(line) or is_comment_line(line):
            continue

        # Parse the data row
        row = parse_data_row_line(line, attributes)

        # Append the row to the data
        data.append(row)

    # Return the data
    return data


def get_data_section_parallel(file, attributes):
    """
    Gets the data from the data section of the ARFF file.

    :param file:        The ARFF file we are processing.
    :param attributes:  The attributes of the ARFF file.
    :return:            The list of data rows.
    """

    # Keep reading lines until we run out of file
    lines = [line for line in file if not (is_whitespace_only(line) or is_comment_line(line))]

    # Parse the lines in parallel
    with Pool(initializer=pool_initialiser_parallel, initargs=(attributes,)) as pool:
        return pool.map(parse_data_row_line_parallel, lines, PARALLEL_BATCH_SIZE)


def pool_initialiser_parallel(attributes):
    """
    Pool initialisation method for when using parallel data reading.

    :param attributes:  The attributes of the ARFF file.
    :return:            Nothing.
    """

    # Make the attributes global so parse_data_row_line_parallel can access them.
    global attributes_global
    attributes_global = attributes


def parse_data_row_line_parallel(line):
    """
    Parallel implementation of parse_data_row_line.

    :param line:        The line of data to parse.
    :return:            A list of data values.
    """

    # Defers to normal implementation by accessing attributes from global
    return parse_data_row_line(line, attributes_global)


def parse_relation_line(line):
    """
    Gets the name of the relation from the line of the ARFF file
    beginning with the @relation keyword.

    :param line:    The line beginning with the @relation keyword.
    :return:        The name of the relation.
    """

    # Save the original line for error messages
    original_line = line

    # Remove the keyword and any following whitespace
    line = remove_keyword(line, RELATION_KEYWORD).lstrip()

    # Extract the relation name
    relation, line = consume(line, NAME_PATTERN)

    # Make sure we got a name back
    if relation is None:
        raise RelationNameNotFoundError(original_line)

    # Make sure the entire line has been consumed
    if not is_whitespace_only(line):
        raise UnrecognisedContentError(line, original_line)

    # Return the relation name
    return remove_quotes(relation)


def parse_attribute_line(line):
    """
    Parses an attribute line from the ARFF file into an attribute object.

    :param line:    The line (beginning with @attribute) to parse.
    :return:        The attribute object.
    """

    # Save the original line for error messages
    original_line = line

    # Remove the @attribute keyword
    line = remove_keyword(line, ATTRIBUTE_KEYWORD)

    # Extract the attribute name
    name, line = consume(line, NAME_PATTERN)

    # Check we found a name
    if name is None:
        raise AttributeNameNotFoundError(original_line)

    # Extract the attribute type
    attribute_type, additional, line = consume_attribute_type_information(line)

    # Check we found a type
    if attribute_type is None:
        raise AttributeTypeNotFoundError(original_line)

    # If the line has not been entirely consumed, raise an error
    if not is_whitespace_only(line):
        raise UnrecognisedContentError(line, original_line)

    # Return the attribute and its type (with any additional info)
    return create_attribute_object(name, attribute_type, additional)


def parse_data_row_line(line, attributes):
    """
    Parses a line of data from the @data section of the ARFF file.

    :param line:        The line of data to parse.
    :param attributes:  The attributes of the ARFF file.
    :return:            A list of data values.
    """

    # Save the original line for error messages
    original_line = line

    # Get the number of expected values
    num_attributes = len(attributes)

    # Create the data row with all missing values
    row = [None] * num_attributes

    # Keep extracting values until we have enough
    i = 0
    j = 0
    line_length = len(line)
    while i < num_attributes:
        # Make sure there is string left to consume
        if j == line_length:
            break

        # Advance through any leading spaces
        while line[j] == ' ':
            j += 1

        # Advance through any quoted section
        if line[j] in QUOTES:
            # Advance till the next matching quote
            quote = line[j]
            j += 1
            start = j
            while line[j] != quote:
                j += 1
            end = j
            j += 2
        else:
            # Advance to the next delimiter
            start = j
            while line[j] not in {'\t', ',', '\n'}:
                j += 1
            end = j
            j += 1

        value = line[start:end]

        # Can skip missing values as the row is initialised with them
        if value == MISSING_VALUE_SYMBOL:
            i += 1
            continue

        # Get the attribute and its type
        attribute = attributes[i]
        attribute_type = attribute['type']

        # Convert numeric types into the defined internal type
        if attribute_type == NUMERIC_KEYWORD:
            value = NUMERIC_TYPE(value)

        elif attribute_type == NOMINAL_KEYWORD:
            # Check the string is one of the allowed values
            if value not in attribute['values']:
                raise InvalidNominalValueError(value, attribute)

        # Add the value to the row
        row[i] = value

        # Increment the index
        i += 1

    # Make sure the line wasn't too long or too short
    if j != line_length or i != num_attributes:
        raise DataSizeMismatchError(num_attributes, original_line)

    return row


def consume_attribute_type_information(line):
    """
    Extracts the type of an attribute from the beginning of a line.

    :param line:    The line to extract the attribute type from.
    :return:        The attribute type, any additional type information, and the remainder of the line.
    """

    # Find the attribute type
    attribute_type, line = consume(line, ATTRIBUTE_TYPE_PATTERN)

    # Abort if no match found
    if attribute_type is None:
        return None, None, line

    # Initialise no additional information
    additional = None

    # Add explicit nominal type
    if attribute_type.startswith('{'):
        additional = attribute_type
        attribute_type = NOMINAL_KEYWORD

    # Lower-case the type
    attribute_type = attribute_type.lower()

    # Keep the date format string as additional
    if attribute_type == DATE_KEYWORD:
        additional, line = consume(line, DATE_FORMAT_PATTERN)
    # Normalise numeric types
    elif attribute_type == REAL_KEYWORD or attribute_type == INTEGER_KEYWORD:
        additional = attribute_type
        attribute_type = NUMERIC_KEYWORD

    # Return the type information and the rest of the line
    return attribute_type, additional, line


def create_attribute_object(name, attribute_type, additional):
    """
    Creates an attribute object for the given attribute information.

    :param name:            The name of the attribute.
    :param attribute_type:  The type of the attribute.
    :param additional:      The additional information about the attribute.
    :return:                A dictionary representing the attribute.
    """

    # Create the dictionary with name and type entries
    attribute_object = {'name': remove_quotes(name),
                        'type': attribute_type}

    # Add the additional information for the given type
    if attribute_type == NOMINAL_KEYWORD:
        attribute_object['values'] = parse_nominal_values(additional)
    elif attribute_type == DATE_KEYWORD:
        attribute_object['format'] = parse_date_format(additional)
    elif attribute_type == NUMERIC_KEYWORD and additional is not None:
        attribute_object['sub-type'] = additional

    # Return the dictionary
    return attribute_object


def parse_nominal_values(string):
    """
    Parses the string of nominal values into a Python list.

    :param string:  The string specification of nominal values (including braces).
    :return:        The list of nominal values.
    """

    # Save the original string for error messages
    original_string = string

    # Create the list
    values = []

    # Remove the leading brace
    string = string[1:]

    # Keep reading values until all are consumed
    while string != '}':
        # Remove any leading whitespace
        string = string.lstrip()

        # Get the name of the next value
        value, string = consume(string, NOMINAL_VALUE_PATTERN)

        # Make sure a value was found
        if value is None:
            raise NominalValuesError(original_string)

        # Put it in the list
        values.append(remove_quotes(value))

        # Remove any trailing whitespace after the value
        string = string.lstrip()

        # Remove the next delimiter
        if string[0] == ',':
            string = string[1:]
        elif string[0] != '}':
            raise NominalValuesError(original_string)

    # Return the list
    return values


def parse_date_format(string):
    """
    Parses the date-format string for date-type attributes.
    TODO: Implement. csterling

    :param string:      The date-format string to parse.
    :return:            Currently, just the input.
    """

    return string


def consume(line, pattern, ignore_leading_whitespace=True, case_insensitive=True):
    """
    Consumes the beginning portion of the line that is matched by the given pattern.

    :param line:                        The line to consume from the start of.
    :param pattern:                     The pattern to match for consumption.
    :param ignore_leading_whitespace:   Whether to drop any leading whitespace.
    :param case_insensitive:            Whether to ignore case.
    :return:                            The consumed part of the line, and the remainder.
    """

    # Remove leading whitespace if ignoring
    if ignore_leading_whitespace:
        line = line.lstrip()

    # Initialise the flags
    flags = re.VERBOSE
    if case_insensitive:
        flags |= re.IGNORECASE

    # Match the pattern
    match = re.match(pattern, line, flags)

    # Return the matched part of the line and the remainder
    if match:
        return line[:match.end()], line[match.end():]
    else:
        return None, line


def read_till_found(file, words, search_comments=False, case_insensitive=True):
    """
    Reads lines from the ARFF file until a line is found that contains at least
    one of the given search words.

    :param file:                The ARFF file to read from.
    :param words:               The set of words to search for.
    :param search_comments:     Whether to allow the search to look inside comments as well.
    :param case_insensitive:    Whether the search should be case-insensitive.
    :return:                    The line where the word was found, or None if no line was found.
    """

    # Lower-case all search words if doing case-insensitive search
    if case_insensitive:
        words = {word.lower() for word in words}

    # Keep reading lines until we run out or find the desired line
    line = None
    while line != "":
        # Read the next line from the file
        line = file.readline()

        # Skip blank lines
        if is_whitespace_only(line):
            continue

        # Skip comments if we are not searching them
        if not search_comments:
            if is_comment_line(line):
                continue

        # Lower-case the searched line if doing case-insensitive search
        search_line = line.lower() if case_insensitive else line

        # If it contains one of the words, return it
        for word in words:
            if search_line.find(word) != -1:
                return line

    # None of the words were found
    raise KeywordNotFoundError(words)


def remove_quotes(string):
    """
    Removes quotes from around a string, if they are present.

    :param string:  The string to remove quotes from.
    :return:        The string without quotes.
    """

    # If starts and ends with double-quotes, remove them
    if string.startswith('\"') and string.endswith('\"'):
        string = string[1:-1]

    # If starts and ends with single-quotes, remove them
    elif string.endswith('\'') and string.endswith('\''):
        string = string[1:-1]

    # Return the (unquoted) string
    return string


def is_comment_line(line):
    """
    Checks whether the given line is a comment line.

    :param line:    The line to check.
    :return:        True if the line is a comment line, False if not.
    """

    return line_starts_with(line, COMMENT_SYMBOL, False)


def remove_keyword(line, keyword):
    """
    Removes the given keyword from the beginning of the line.

    :param line:        The line to remove the keyword from.
    :param keyword:     The keyword to remove.
    :return:            The line with the keyword removed.
    """

    # The line should start with the given keyword
    assert line_starts_with(line, keyword), line + ' does not start with keyword: ' + keyword

    # Return the line with the keyword removed
    return line[len(keyword):]


def is_whitespace_only(line):
    """
    Checks if the given line contains only whitespace.

    :param line:    The line to check.
    :return:        True if the line contains only whitespace, False if not.
    """

    return line.strip() == ''


def line_starts_with(line, string, case_insensitive=True):
    """
    Checks if the line starts with the given string.

    :param line:                The line to check.
    :param string:              The string to look for.
    :param case_insensitive:    Whether the check should disregard case.
    :return:                    True if the line starts with the given string, False otherwise.
    """

    # Remove case information if case-insensitive
    if case_insensitive:
        line = line.lower()
        string = string.lower()

    # Return the result of the check
    return line.startswith(string)


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


class ARFFFileError(Exception):
    """
    Base class for all exceptions thrown when reading ARFF files.
    """

    def __init__(self, message):
        super(ARFFFileError, self).__init__(message)


class KeywordNotFoundError(ARFFFileError):
    """
    Exception where an expected keyword was never found in the ARFF file.
    """

    def __init__(self, keywords):
        # Format the message
        message = 'Couldn\'t find line containing any of: ' + self.keywords_to_string(keywords)
        super(KeywordNotFoundError, self).__init__(message)

    @staticmethod
    def keywords_to_string(keywords):
        return '{' + ','.join(keywords) + '}'


class UnrecognisedContentError(ARFFFileError):
    """
    Exception when part of a line cannot be parsed because it is not valid ARFF content.
    """

    def __init__(self, content, line):
        message = 'Unrecognised content \'' + content + '\' in line: ' + line
        super(UnrecognisedContentError, self).__init__(message)


class RelationNameNotFoundError(ARFFFileError):
    """
    Exception for when the name of the relation cannot be found.
    """

    def __init__(self, line):
        message = 'Couldn\'t parse relation name from: ' + line
        super(RelationNameNotFoundError, self).__init__(message)


class AttributeNameNotFoundError(ARFFFileError):
    """
    Exception when a name for an attribute cannot be parsed.
    """

    def __init__(self, line):
        message = 'Couldn\'t parse attribute name from: ' + line
        super(AttributeNameNotFoundError, self).__init__(message)


class AttributeTypeNotFoundError(ARFFFileError):
    """
    Exception for when an attribute's type cannot be determined.
    """

    def __init__(self, line):
        message = 'Couldn\'t determine attribute type from: ' + line
        super(AttributeTypeNotFoundError, self).__init__(message)


class DataSizeMismatchError(ARFFFileError):
    """
    Exception when there are too many or too few values in a given data row.
    """

    def __init__(self, num_attributes, line):
        message = 'Wrong number of values (require ' + str(num_attributes) + ') in: ' + line
        super(DataSizeMismatchError, self).__init__(message)


class NominalValuesError(ARFFFileError):
    """
    Exception for when the set of nominal values can't be parsed.
    """

    def __init__(self, string):
        message = 'Error parsing nominal values from: ' + string
        super(NominalValuesError, self).__init__(message)


class InvalidNominalValueError(ARFFFileError):
    """
    Exception for when a nominal data value isn't one of the pre-specified
    values from the attribute declaration.
    """

    def __init__(self, value, attribute):
        message = 'Illegal nominal value for attribute \'' + attribute['name'] + '\': ' + value + '\n'
        message += 'Allowed values are: ' + str(attribute['values'])
        super(InvalidNominalValueError, self).__init__(message)
