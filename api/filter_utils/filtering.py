# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""API response filtering modules"""


def apply(args, data):
    """Filter results based on the arguments provided"""
    filter_sort = args.get('sort')
    filter_name = args.get('name')
    filter_type = args.get('type')
    filter_arch = args.get('network_arch')
    filter_read_only = args.get('read_only')
    filter_format = args.get('format')
    filter_tag = args.get('tag')

    if filter_name is not None:
        if filter_name.startswith('!'):
            filter_name = filter_name[1:]
            data = list(filter(lambda d: d.get('name') != filter_name, data))
        else:
            data = list(filter(lambda d: d.get('name') == filter_name, data))
    if filter_type is not None:
        if filter_type.startswith('!'):
            filter_type = filter_type[1:]
            data = list(filter(lambda d: d.get('type') != filter_type, data))
        else:
            data = list(filter(lambda d: d.get('type') == filter_type, data))
    if filter_arch is not None:
        if filter_arch.startswith('!'):
            filter_arch = filter_arch[1:]
            data = list(filter(lambda d: d.get('network_arch') != filter_arch, data))
        else:
            data = list(filter(lambda d: d.get('network_arch') == filter_arch, data))
    if filter_read_only is not None:
        filter_read_only_as_boolean = filter_read_only == 'true'
        data = list(filter(lambda d: d.get('read_only') == filter_read_only_as_boolean, data))

    if filter_format is not None:
        if filter_format.startswith('!'):
            filter_format = filter_format[1:]
            data = list(filter(lambda d: d.get('format') != filter_format, data))
        else:
            data = list(filter(lambda d: d.get('format') == filter_format, data))

    if filter_tag is not None:
        if filter_tag.startswith('!'):
            filter_tag = filter_tag[1:]
            data = list(filter(lambda d: filter_tag not in d.get('tags', []), data))
        else:
            data = list(filter(lambda d: filter_tag in d.get('tags', []), data))

    if filter_sort == 'name-ascending':
        data = sorted(data, key=lambda d: '' + d.get('name') + ':' + d.get('version'), reverse=False)
    elif filter_sort == 'name-descending':
        data = sorted(data, key=lambda d: '' + d.get('name') + ':' + d.get('version'), reverse=True)
    elif filter_sort == 'date-ascending':
        data = sorted(data, key=lambda d: d.get('last_modified'), reverse=False)
    else:  # filter_sort == 'date-descending'
        data = sorted(data, key=lambda d: d.get('last_modified'), reverse=True)

    return data
