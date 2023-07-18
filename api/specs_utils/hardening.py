# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Json spec schema hardening modules"""
import copy

from specs_utils import csv_to_json_schema

from jsonschema import validate as validationDriver, exceptions


def __merge(d1, d2):
    """Merge 2 dictionaries"""
    for key in d2.keys():
        if key not in d1:
            d1[key] = d2[key]
        elif d1[key] is None:
            d1[key] = d2[key]
        elif type(d1[key]) is list and type(d2[key]) is list:
            if d1[key] != [] and type(d1[key][0]) is dict:
                for i in range(0, min(len(d1[key]), len(d2[key]))):
                    __merge(d1[key][i], d2[key][i])
            else:
                d1[key] = d1[key] + [i for i in d2[key] if i not in d1[key]]
        elif type(d2[key]) is not dict:
            d1[key] = d2[key]
        else:
            __merge(d1[key], d2[key])
    return d1


def harden(data, schema):
    """Harden the schema provided"""
    return __merge(copy.deepcopy(schema['default']), data)


def validate(data, schema):
    """Validate the schema provided"""
    try:
        validationDriver(instance=data, schema=schema)
    except exceptions.ValidationError as e:
        return e.message
    return None


# test code
if __name__ == '__main__':
    schema = csv_to_json_schema.convert("specs/detectnet_v2/detectnet_v2 - train.csv")

    # positive test
    hardened_data = harden(data={'random_seed': 99}, schema=schema)
    err = validate(data=hardened_data, schema=schema)
    if err:
        print(err)

    # negative test
    err = validate(data={'random_seed': 99}, schema=schema)
    if err:
        print(err)
