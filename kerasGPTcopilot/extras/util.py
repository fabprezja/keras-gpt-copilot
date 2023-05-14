# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

from json import JSONEncoder
import numpy as np


class CustomJSONEncoder(JSONEncoder):
    """
    A custom JSON encoder that converts NumPy float32 objects to native Python floats.

    This class inherits from the built-in JSONEncoder class and overrides the default method to handle
    NumPy float32 objects, which are not natively serializable by the default JSON encoding.

    Usage:
        Use this class as a custom encoder for the json.dumps() function by passing it as the `cls` argument.

    Example:
        import json
        from CustomJSONEncoder import CustomJSONEncoder
        data = {'key': np.float32(3.14)}
        json_data = json.dumps(data, cls=CustomJSONEncoder)

    Attributes:
        None
    """
    def default(self, obj):
        """
        Overridden default method that handles NumPy float32 objects by converting them to native Python floats.

        Args:
            obj (Any): The object to be serialized.

        Returns:
            float: The native Python float representation of the NumPy float32 object
            Any: The result of calling the default method of the JSONEncoder superclass
        """
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)
