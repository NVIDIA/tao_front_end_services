#!/usr/bin/env python3

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

"""API modules defining schemas and endpoints"""
import ast
import sys
import math
import json
import shutil
import atexit
import os
import threading
import time

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from flask import Flask, request, jsonify, make_response, render_template, send_from_directory, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from marshmallow import Schema, fields, exceptions, validate
from marshmallow_enum import EnumField, Enum

from filter_utils import filtering, pagination
from auth_utils import credentials, authentication, access_control
from health_utils import health_check

from handlers.app_handler import AppHandler as app_handler
from handlers.ngc_handler import mount_ngc_workspace, unmount_ngc_workspace, workspace_info, load_user_workspace_metadata, is_ngc_workspace_free
from handlers.stateless_handlers import safe_dump_file, resolve_metadata
from handlers.utilities import validate_uuid
from utils.utils import run_system_command, is_pvc_space_free
from job_utils.workflow import Workflow, report_healthy, synchronized

from werkzeug.exceptions import HTTPException
from datetime import timedelta
from timeloop import Timeloop
from functools import wraps

flask_plugin = FlaskPlugin()
marshmallow_plugin = MarshmallowPlugin()


#
# Utils
#
def sys_int_format():
    """Get integer format based on system."""
    if sys.maxsize > 2**31 - 1:
        return "int64"
    return "int32"


def disk_space_check(f):
    """Decorator to check disk space for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = kwargs.get('user_id', '')
        threshold_bytes = 100 * 1024 * 1024

        pvc_free_space, pvc_free_bytes = is_pvc_space_free(threshold_bytes)
        msg = f"PVC free space remaining is {pvc_free_bytes} bytes which is less than {threshold_bytes} bytes"
        if not pvc_free_space:
            return make_response(jsonify({'error': f'Disk space is nearly full. {msg}. Delete appropriate experiments/datasets'}), 500)

        if os.getenv("NGC_RUNNER") == "True" and user_id:
            workspace_free_space, workspace_free_bytes = is_ngc_workspace_free(user_id, threshold_bytes)
            msg = f"NGC workspace free space remaining is {workspace_free_bytes} bytes which is less than {threshold_bytes} bytes"
            if not workspace_free_space:
                return make_response(jsonify({'error': f'Disk space is nearly full. {msg}. Delete appropriate experiments/datasets'}), 500)

        return f(*args, **kwargs)

    return decorated_function


#
# Create an APISpec
#
spec = APISpec(
    title='NVIDIA Transfer Learning API',
    version='v5.2.0',
    openapi_version='3.0.3',
    info={"description": 'NVIDIA Transfer Learning API document'},
    tags=[
        {"name": 'AUTHENTICATION', "description": 'Endpoints related to User Authentication'},
        {"name": 'DATASET', "description": 'Endpoints related to Datasets'},
        {"name": 'EXPERIMENT', "description": 'Endpoints related to Experiments'},
        {"name": "nSpectId", "description": "NSPECT-1T59-RTYH", "externalDocs": {"url": "https://nspect.nvidia.com/review?id=NSPECT-1T59-RTYH"}}
    ],
    plugins=[flask_plugin, marshmallow_plugin],
    security=[{"bearer-token": []}],
)

api_key_scheme = {"type": "apiKey", "in": "header", "name": "ngc_api_key"}
jwt_scheme = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT", "description": "RFC8725 Compliant JWT"}

spec.components.security_scheme("api-key", api_key_scheme)
spec.components.security_scheme("bearer-token", jwt_scheme)

spec.components.header("X-RateLimit-Limit", {
    "description": "The number of allowed requests in the current period",
    "schema": {
        "type": "integer",
        "format": sys_int_format(),
        "minimum": -sys.maxsize - 1,
        "maximum": sys.maxsize,
    }
})


#
# Enum stuff for APISpecs
#
def enum_to_properties(self, field, **kwargs):
    """
    Add an OpenAPI extension for marshmallow_enum.EnumField instances
    """
    if isinstance(field, EnumField):
        return {'type': 'string', 'enum': [m.name for m in field.enum]}
    return {}


marshmallow_plugin.converter.add_attribute_function(enum_to_properties)


#
# Global schemas and enums
#
class DatasetUploadSchema(Schema):
    """Class defining dataset upload schema"""

    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))


class ErrorRspSchema(Schema):
    """Class defining error response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    error_desc = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    error_code = fields.Int(validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize), format=sys_int_format())


class JobStatusEnum(Enum):
    """Class defining job status enum"""

    Done = 'Done'
    Running = 'Running'
    Error = 'Error'
    Pending = 'Pending'
    Canceled = 'Canceled'


class PullStatus(Enum):
    """Class defining artifact upload/download status"""

    not_present = "not_present"
    in_progress = "in_progress"
    present = "present"


#
# Flask app
#
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['TRAP_HTTP_EXCEPTIONS'] = True
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["10000/hour"],
    headers_enabled=True,
    storage_uri="memory://",
)


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


#
# JobResultSchema
#
class DetailedStatusSchema(Schema):
    """Class defining Status schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    date = fields.Str(format="mm/dd/yyyy", validate=fields.validate.Length(max=26))
    time = fields.Str(format="hh:mm:ss", validate=fields.validate.Length(max=26))
    message = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    status = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class GraphSchema(Schema):
    """Class defining Graph schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    x_min = fields.Int(allow_none=True, validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize), format=sys_int_format())
    x_max = fields.Int(allow_none=True, validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize), format=sys_int_format())
    y_min = fields.Float(allow_none=True)
    y_max = fields.Float(allow_none=True)
    values = fields.Dict(keys=fields.Int(allow_none=True), values=fields.Float(allow_none=True))
    units = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class CategoryWiseSchema(Schema):
    """Class defining CategoryWise schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    category = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    value = fields.Float(allow_none=True)


class CategorySchema(Schema):
    """Class defining Category schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    category_wise_values = fields.List(fields.Nested(CategoryWiseSchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))


class KPISchema(Schema):
    """Class defining KPI schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    values = fields.Dict(keys=fields.Int(allow_none=True), values=fields.Float(allow_none=True))


class CustomFloatField(fields.Float):
    """Class defining custom Float field allown NaN and Inf values in Marshmallow"""

    def _deserialize(self, value, attr, data, **kwargs):
        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        if value == "inf" or (isinstance(value, float) and math.isinf(value)):
            return float("inf")
        if value == "-inf" or (isinstance(value, float) and math.isinf(value)):
            return float("-inf")
        if value is None:
            return None
        return super()._deserialize(value, attr, data)


class AutoMLResultsSchema(Schema):
    """Class defining AutoML results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    value = CustomFloatField(allow_none=True)


class StatsSchema(Schema):
    """Class defining results stats schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    value = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))


class JobResultSchema(Schema):
    """Class defining job results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    detailed_status = fields.Nested(DetailedStatusSchema, allow_none=True)
    graphical = fields.List(fields.Nested(GraphSchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    categorical = fields.List(fields.Nested(CategorySchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    kpi = fields.List(fields.Nested(KPISchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    automl_result = fields.List(fields.Nested(AutoMLResultsSchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    stats = fields.List(fields.Nested(StatsSchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    epoch = fields.Int(allow_none=True, validate=fields.validate.Range(min=-1, max=sys.maxsize), format=sys_int_format(), error="Epoch should be larger than -1. With -1 meaning non-valid.")
    max_epoch = fields.Int(allow_none=True, validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format(), error="Max epoch should be non negative.")
    time_per_epoch = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    time_per_iter = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    cur_iter = fields.Int(allow_none=True, validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    eta = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    key_metric = fields.Float(allow_none=True)
    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))


class AllowedDockerEnvVariables(Enum):
    """Allowed docker environment variables while launching DNN containers"""

    WANDB_API_KEY = "WANDB_API_KEY"
    CLEARML_WEB_HOST = "CLEARML_WEB_HOST"
    CLEARML_API_HOST = "CLEARML_API_HOST"
    CLEARML_FILES_HOST = "CLEARML_FILES_HOST"
    CLEARML_API_ACCESS_KEY = "CLEARML_API_ACCESS_KEY"
    CLEARML_API_SECRET_KEY = "CLEARML_API_SECRET_KEY"


#
# AUTHENTICATION API
#
class LoginReqSchema(Schema):
    """Class defining login request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    ngc_api_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))


class LoginRspSchema(Schema):
    """Class defining login response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    token = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))


@app.route('/api/v1/login', methods=['POST'])
@disk_space_check
def login():
    """User Login.
    ---
    post:
      tags:
      - AUTHENTICATION
      summary: User Login
      description: Returns the user credentials
      security:
        - api-key: []
      requestBody:
        content:
          application/json:
            schema: LoginReqSchema
        description: Login request with ngc_api_key
        required: true
      responses:
        201:
          description: Retuned the new Dataset
          content:
            application/json:
              schema: LoginRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        401:
          description: Unauthorized
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = LoginReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    key = request_dict.get('ngc_api_key', 'invalid_key')
    creds, err = credentials.get_from_ngc(key)
    if err:
        print("Unauthorized: " + err, flush=True)
        metadata = {"error_desc": "Unauthorized: " + err, "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 401)
    schema = LoginRspSchema()
    schema_dict = schema.dump(schema.load(creds))
    return make_response(jsonify(schema_dict), 201)


# Internal endpoint for ingress controller to check authentication
@app.route('/api/v1/auth', methods=['GET'])
@disk_space_check
def auth():
    """authentication endpoint"""
    # retrieve jwt from headers
    token = ''
    url = request.headers.get('X-Original-Url', '')
    print('URL: ' + str(url), flush=True)
    method = request.headers.get('X-Original-Method', '')
    print('Method: ' + str(method), flush=True)
    # bypass authentication for http OPTIONS requests
    if method == 'OPTIONS':
        return make_response(jsonify({}), 200)
    # retrieve authorization token, or use NGC SID/SSID cookie
    authorization = request.headers.get('Authorization', '')
    authorization_parts = authorization.split()
    if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'bearer':
        token = authorization_parts[1]
    if not token:
        sid_cookie = request.cookies.get('SID')
        if sid_cookie:
            token = 'SID=' + sid_cookie
    if not token:
        ssid_cookie = request.cookies.get('SSID')
        if ssid_cookie:
            token = 'SSID=' + ssid_cookie
    print('Token: ...' + str(token)[-10:], flush=True)
    # authentication
    user_id, err = authentication.validate(token)
    if err:
        print("Unauthorized: " + str(err), flush=True)
        metadata = {"error_desc": str(err), "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
        return response
    # access control
    err = access_control.validate(url, user_id)
    if err:
        print("Forbidden: " + str(err), flush=True)
        metadata = {"error_desc": str(err), "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 403)
        return response
    return make_response(jsonify({'user_id': user_id}), 200)


# Define enum common to Dataset and Experiment Api

class ActionEnum(Enum):
    """Class defining action type enum"""

    dataset_convert = 'dataset_convert'
    convert = 'convert'
    convert_and_index = 'convert_and_index'
    convert_efficientdet_tf1 = 'convert_efficientdet_tf1'
    convert_efficientdet_tf2 = 'convert_efficientdet_tf2'

    kmeans = 'kmeans'
    augment = 'augment'

    train = 'train'
    evaluate = 'evaluate'
    prune = 'prune'
    retrain = 'retrain'
    export = 'export'
    gen_trt_engine = 'gen_trt_engine'
    trtexec = 'trtexec'
    inference = 'inference'

    annotation = 'annotation'
    analyze = 'analyze'
    validate = 'validate'
    generate = 'generate'
    calibration_tensorfile = 'calibration_tensorfile'
    confmat = 'confmat'
    nextimage = 'nextimage'
    cacheimage = 'cacheimage'
    notify = 'notify'

    auto3dseg = 'auto3dseg'

#
# DATASET API
#


class DatasetActions(Schema):
    """Class defining dataset actions schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()


class DatasetTypeEnum(Enum):
    """Class defining dataset type enum"""

    object_detection = 'object_detection'
    semantic_segmentation = 'semantic_segmentation'
    image_classification = 'image_classification'
    instance_segmentation = 'instance_segmentation'
    character_recognition = 'character_recognition'
    bpnet = 'bpnet'
    fpenet = 'fpenet'
    action_recognition = 'action_recognition'
    pointpillars = 'pointpillars'
    pose_classification = 'pose_classification'
    ml_recog = 'ml_recog'
    ocdnet = 'ocdnet'
    ocrnet = 'ocrnet'
    optical_inspection = 'optical_inspection'
    re_identification = 're_identification'
    visual_changenet = 'visual_changenet'
    centerpose = 'centerpose'
    user_custom = 'user_custom'


class DatasetFormatEnum(Enum):
    """Class defining dataset format enum"""

    kitti = 'kitti'
    pascal_voc = 'pascal_voc'
    raw = 'raw'
    coco_raw = 'coco_raw'
    unet = 'unet'
    coco = 'coco'
    lprnet = 'lprnet'
    default = 'default'
    custom = 'custom'
    classification_pyt = 'classification_pyt'
    visual_changenet_segment = 'visual_changenet_segment'
    visual_changenet_classify = 'visual_changenet_classify'
    medical = 'medical'


class JobTarStatsSchema(Schema):
    """Class defining job tar file stats schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    sha256_digest = fields.Str(format="regex", regex=r'^[a-fA-F0-9]{64}$', validate=fields.validate.Length(max=64), allow_none=True)
    file_size = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)


class DatasetReqSchema(Schema):
    """Class defining dataset request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    logo = fields.URL(validate=fields.validate.Length(max=2048))
    type = EnumField(DatasetTypeEnum)
    format = EnumField(DatasetFormatEnum)
    pull = fields.URL(validate=fields.validate.Length(max=2048))
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)


class DatasetJobSchema(Schema):
    """Class defining dataset job result total schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    created_on = fields.Str(format='date-time', validate=fields.validate.Length(max=26))
    last_modified = fields.Str(format='date-time', validate=fields.validate.Length(max=26))
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    result = fields.Nested(JobResultSchema)
    specs = fields.Raw(allow_none=True)
    job_tar_stats = fields.Nested(JobTarStatsSchema)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class DatasetRspSchema(Schema):
    """Class defining dataset response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        load_only = ("client_id", "client_secret", "filters")

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = fields.Str(format='date-time', validate=fields.validate.Length(max=26))
    last_modified = fields.Str(format='date-time', validate=fields.validate.Length(max=26))
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    type = EnumField(DatasetTypeEnum)
    format = EnumField(DatasetFormatEnum)
    pull = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    actions = fields.List(EnumField(ActionEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    jobs = fields.List(fields.Nested(DatasetJobSchema), validate=validate.Length(max=sys.maxsize))
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)


class DatasetListRspSchema(Schema):
    """Class defining dataset list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    datasets = fields.List(fields.Nested(DatasetRspSchema), validate=validate.Length(max=sys.maxsize))


class DatasetJobListSchema(Schema):
    """Class defining dataset list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    jobs = fields.List(fields.Nested(DatasetJobSchema), validate=validate.Length(max=sys.maxsize))


@app.route('/api/v1/users/<user_id>/datasets', methods=['GET'])
@disk_space_check
def dataset_list(user_id):
    """List Datasets.
    ---
    get:
      tags:
      - DATASET
      summary: List Datasets
      description: Returns the list of Datasets
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Optional name filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: format
        in: query
        description: Optional format filter
        required: false
        schema:
          type: string
          enum: ["medical", "unet", "custom" ]
      - name: type
        in: query
        description: Optional type filter
        required: false
        schema:
          type: string
          enum: [ "object_detection", "semantic_segmentation", "image_classification" ]
      responses:
        200:
          description: Returned list of Datasets
          content:
            application/json:
              schema:
                type: array
                items: DatasetRspSchema
                maxItems: 2147483647
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    datasets = app_handler.list_datasets(user_id)
    filtered_datasets = filtering.apply(request.args, datasets)
    paginated_datasets = pagination.apply(request.args, filtered_datasets)
    pagination_total = len(filtered_datasets)
    metadata = {"datasets": paginated_datasets}
    schema = DatasetListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))['datasets']))
    response.headers['X-Pagination-Total'] = str(pagination_total)
    return response


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>', methods=['GET'])
@disk_space_check
def dataset_retrieve(user_id, dataset_id):
    """Retrieve Dataset.
    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Dataset
      description: Returns the Dataset
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID of Dataset to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.retrieve_dataset(user_id, dataset_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>', methods=['DELETE'])
@disk_space_check
def dataset_delete(user_id, dataset_id):
    """Delete Dataset.
    ---
    delete:
      tags:
      - DATASET
      summary: Delete Dataset
      description: Cancels all related running jobs and returns the deleted Dataset
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID of Dataset to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Deleted Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_dataset(user_id, dataset_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets', methods=['POST'])
@disk_space_check
def dataset_create(user_id):
    """Create new Dataset.
    ---
    post:
      tags:
      - DATASET
      summary: Create new Dataset
      description: Returns the new Dataset
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetReqSchema
        description: Initial metadata for new Dataset (type and format required)
        required: true
      responses:
        201:
          description: Retuned the new Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.create_dataset(user_id, request_dict)
    # Get schema
    schema = None
    if response.code == 201:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>', methods=['PUT'])
@disk_space_check
def dataset_update(user_id, dataset_id):
    """Update Dataset.
    ---
    put:
      tags:
      - DATASET
      summary: Update Dataset
      description: Returns the updated Dataset
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID of Dataset to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetReqSchema
        description: Updated metadata for Dataset
        required: true
      responses:
        200:
          description: Returned the updated Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_dataset(user_id, dataset_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>', methods=['PATCH'])
@disk_space_check
def dataset_partial_update(user_id, dataset_id):
    """Partial update Dataset.
    ---
    patch:
      tags:
      - DATASET
      summary: Partial update Dataset
      description: Returns the updated Dataset
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID of Dataset to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetReqSchema
        description: Updated metadata for Dataset
        required: true
      responses:
        200:
          description: Returned the updated Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_dataset(user_id, dataset_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route("/api/v1/users/<user_id>/datasets/<dataset_id>:upload", methods=["POST"])
@disk_space_check
def dataset_upload(user_id, dataset_id):
    """Upload Dataset.
    ---
    post:
      tags:
      - DATASET
      summary: Upload Dataset
      description: Upload training and testing data
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        description: Data file to upload (a tar.gz file)
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                data:
                  type: string
                  format: binary
                  maxLength: 5000
            encoding:
              data:
                contentType: application/gzip
        required: true
      responses:
        201:
          description: Upload sucessful
          content:
            application/json:
              schema: DatasetUploadSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    file_tgz = request.files.get("file", None)

    if file_tgz:
        file_tgz.seek(0, 2)
        file_size = file_tgz.tell()
        print("Dataset size", file_size, file=sys.stderr)
        max_file_size = 250 * 1024 * 1024
        if file_size > max_file_size:
            return make_response(jsonify({'error': 'File size exceeds the limit of 250 MB; Use Dataset pull feature for such datasets'}), 400)
        file_tgz.seek(0)

    # Get response
    print("Triggering API call to upload data to server", file=sys.stderr)
    response = app_handler.upload_dataset(user_id, dataset_id, file_tgz)
    print("API call to upload data to server complete", file=sys.stderr)
    # Get schema
    schema_dict = None
    if response.code == 201:
        schema = DatasetUploadSchema()
        print("Returning success response", file=sys.stderr)
        schema_dict = schema.dump({"message": "Server recieved file and upload process started"})
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
        schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def dataset_specs_schema(user_id, dataset_id, action):
    """Retrieve Specs schema.
    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Specs schema
      description: Returns the Specs schema for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [ "dataset_convert", "convert", "convert_and_index", "convert_efficientdet_tf1", "convert_efficientdet_tf2", "kmeans", "augment", "train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference", "annotation", "analyze", "validate", "generate", "calibration_tensorfile", "confmat" ]
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_spec_schema(user_id, dataset_id, action, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/jobs', methods=['POST'])
@disk_space_check
def dataset_job_run(user_id, dataset_id):
    """Run Dataset Jobs.
    ---
    post:
      tags:
      - DATASET
      summary: Run Dataset Jobs
      description: Asynchronously starts a dataset action and returns corresponding Job ID
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetActions
      responses:
        201:
          description: Returned the Job ID corresponding to requested Dataset Action
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = DatasetActions()
    request_schema_data = schema.dump(schema.load(request_data))
    requested_job = request_schema_data.get('parent_job_id', None)
    if requested_job:
        requested_job = str(requested_job)
    requested_action = request_schema_data.get('action', "")
    specs = request_schema_data.get('specs', {})
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    # Get response
    response = app_handler.job_run(user_id, dataset_id, requested_job, requested_action, "dataset", specs=specs, name=name, description=description)
    handler_metadata = resolve_metadata(user_id, "dataset", dataset_id)
    dataset_format = handler_metadata.get("format")
    # Get schema
    if response.code == 201:
        # MEDICAL dataset jobs are sync jobs and the response should be returned directly.
        if dataset_format == "medical":
            return make_response(jsonify(response.data), response.code)
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/jobs', methods=['GET'])
@disk_space_check
def dataset_job_list(user_id, dataset_id):
    """List Jobs for Dataset.
    ---
    get:
      tags:
      - DATASET
      summary: List Jobs for Dataset
      description: Returns the list of Jobs
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          pattern: '.*'
          maxLength: 36
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending" ]
      responses:
        200:
          description: Returned list of Jobs
          content:
            application/json:
              schema:
                type: array
                items: DatasetJobSchema
                maxItems: 2147483647
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=None if dataset_id in ("*", "all") else dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_list(user_id, dataset_id, "dataset")
    # Get schema
    if response.code == 200:
        pagination_total = 0
        metadata = {"jobs": response.data}
        schema = DatasetJobListSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))['jobs']))
        response.headers['X-Pagination-Total'] = str(pagination_total)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def dataset_job_retrieve(user_id, dataset_id, job_id):
    """Retrieve Job for Dataset.
    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Job for Dataset
      description: Returns the Job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job
          content:
            application/json:
              schema: DatasetJobSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_retrieve(user_id, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetJobSchema()
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def dataset_job_cancel(user_id, dataset_id, job_id):
    """Cancel Dataset Job.
    ---
    post:
      tags:
      - DATASET
      summary: Cancel Dataset Job
      description: Cancel Dataset Job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested cancelation of specified Job ID (asynchronous)
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_cancel(user_id, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def dataset_job_delete(user_id, dataset_id, job_id):
    """Delete Dataset Job.
    ---
    delete:
      tags:
      - DATASET
      summary: Delete Dataset Job
      description: delete Dataset Job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested deletion of specified Job ID
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_delete(user_id, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def dataset_job_files_list(user_id, dataset_id, job_id):
    """List Job Files.
    ---
    get:
      tags:
      - DATASET
      summary: List Job Files
      description: List the Files produced by a given job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Files
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
                  maxLength: 1000
                  maxLength: 1000
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    retrieve_logs = ast.literal_eval(request.args.get("retrieve_logs", "False"))
    retrieve_specs = ast.literal_eval(request.args.get("retrieve_specs", "False"))
    response = app_handler.job_list_files(user_id, dataset_id, job_id, retrieve_logs, retrieve_specs, "dataset")
    # Get schema
    if response.code == 200:
        if isinstance(response.data, list) and (all(isinstance(f, str) for f in response.data) or response.data == []):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: file list invalid", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/jobs/<job_id>:download_selective_files', methods=['GET'])
@disk_space_check
def dataset_job_download_selective_files(user_id, dataset_id, job_id):
    """Download selective Job Artifacts.
    ---
    get:
      tags:
      - DATASET
      summary: Download selective Job Artifacts
      description: Download selective Artifacts produced by a given job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 5000
                maxLength: 5000
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    file_lists = request.args.getlist('file_lists')
    tar_files = ast.literal_eval(request.args.get('tar_files', "True"))
    if not file_lists:
        return make_response(jsonify("No files passed in list format to download or"), 400)
    # Get response
    response = app_handler.job_download(user_id, dataset_id, job_id, "dataset", file_lists=file_lists, tar_files=tar_files)
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/datasets/<dataset_id>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def dataset_job_download(user_id, dataset_id, job_id):
    """Download Job Artifacts.
    ---
    get:
      tags:
      - DATASET
      summary: Download Job Artifacts
      description: Download the Artifacts produced by a given job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
                maxLength: 1000
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_download(user_id, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


#
# EXPERIMENT API
#
class ExperimentActions(Schema):
    """Class defining experiment actions schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()


class CheckpointChooseMethodEnum(Enum):
    """Class defining enum for methods of picking a trained checkpoint"""

    latest_model = 'latest_model'
    best_model = 'best_model'
    from_epoch_number = 'from_epoch_number'


class ExperimentTypeEnum(Enum):
    """Class defining type of experiment"""

    vision = 'vision'
    medical = 'medical'


class ExperimentExportTypeEnum(Enum):
    """Class defining model export type"""

    tao = 'tao'
    medical_bundle = 'medical_bundle'


class ExperimentNetworkArchEnum(Enum):
    """Class defining model network architecure enum"""

    # OD Networks tf
    detectnet_v2 = 'detectnet_v2'
    faster_rcnn = 'faster_rcnn'
    yolo_v4 = 'yolo_v4'
    yolo_v4_tiny = 'yolo_v4_tiny'
    yolo_v3 = 'yolo_v3'
    ssd = 'ssd'
    dssd = 'dssd'
    retinanet = 'retinanet'
    # Other tf networks
    unet = 'unet'
    lprnet = 'lprnet'
    classification_tf1 = 'classification_tf1'
    classification_tf2 = 'classification_tf2'
    efficientdet_tf1 = 'efficientdet_tf1'
    efficientdet_tf2 = 'efficientdet_tf2'
    mask_rcnn = 'mask_rcnn'
    multitask_classification = 'multitask_classification'
    # DriveIX networks
    bpnet = 'bpnet'
    fpenet = 'fpenet'
    # PyT CV networks
    action_recognition = 'action_recognition'
    classification_pyt = 'classification_pyt'
    mal = 'mal'
    ml_recog = 'ml_recog'
    ocdnet = 'ocdnet'
    ocrnet = 'ocrnet'
    optical_inspection = 'optical_inspection'
    pointpillars = 'pointpillars'
    pose_classification = 'pose_classification'
    re_identification = 're_identification'
    deformable_detr = 'deformable_detr'
    dino = 'dino'
    segformer = 'segformer'
    visual_changenet = 'visual_changenet'
    centerpose = 'centerpose'
    # Data analytics networks
    annotations = "annotations"
    analytics = "analytics"
    augmentation = "augmentation"
    auto_label = "auto_label"
    medical_vista3d = "medical_vista3d"
    medical_segmentation = "medical_segmentation"
    medical_annotation = "medical_annotation"
    medical_classification = "medical_classification"
    medical_detection = "medical_detection"
    medical_automl = "medical_automl"
    medical_custom = "medical_custom"


class AutoMLAlgorithm(Enum):
    """Class defining automl algorithm enum"""

    bayesian = "bayesian"
    hyperband = "hyperband"


class ExperimentReqSchema(Schema):
    """Class defining experiment request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    version = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    logo = fields.URL(validate=fields.validate.Length(max=2048))
    ngc_path = fields.Str(format="regex", regex=r'^\w+(/[\w-]+)?/[\w-]+:[\w.-]+$', validate=fields.validate.Length(max=250))
    sha256_digest = fields.Dict(allow_none=True)
    is_ptm_backbone = fields.Bool()
    base_experiment_pull_complete = EnumField(PullStatus)
    additional_id_info = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(keys=fields.Str(format="regex", regex=r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$', validate=fields.validate.Length(max=100), allow_none=True), values=fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True))
    encryption_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    network_arch = EnumField(ExperimentNetworkArchEnum)
    base_experiment = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36)), validate=validate.Length(max=2))
    eval_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    inference_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    calibration_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    train_datasets = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36)), validate=validate.Length(max=sys.maxsize))
    read_only = fields.Bool()
    public = fields.Bool()
    automl_enabled = fields.Bool(allow_none=True)
    automl_algorithm = EnumField(AutoMLAlgorithm, allow_none=True)
    automl_max_recommendations = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_delete_intermediate_ckpt = fields.Bool(allow_none=True)
    override_automl_disabled_params = fields.Bool(allow_none=True)
    automl_R = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_nu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    epoch_multiplier = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_add_hyperparameters = fields.Str(format="regex", regex=r'\[.*\]', validate=fields.validate.Length(max=5000), allow_none=True)
    automl_remove_hyperparameters = fields.Str(format="regex", regex=r'\[.*\]', validate=fields.validate.Length(max=5000), allow_none=True)
    type = EnumField(ExperimentTypeEnum, default=ExperimentTypeEnum.vision)
    realtime_infer = fields.Bool(default=False)
    model_params = fields.Dict(allow_none=True)
    bundle_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    realtime_infer_request_timeout = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)


class ExperimentJobSchema(Schema):
    """Class defining experiment job schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    created_on = fields.Str(format='date-time', validate=fields.validate.Length(max=26))
    last_modified = fields.Str(format='date-time', validate=fields.validate.Length(max=26))
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    result = fields.Nested(JobResultSchema)
    sync = fields.Bool()
    specs = fields.Raw(allow_none=True)
    job_tar_stats = fields.Nested(JobTarStatsSchema)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    experiment_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class ExperimentRspSchema(Schema):
    """Class defining experiment response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        load_only = ("realtime_infer_endpoint", "realtime_infer_model_name")

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = fields.Str(format='date-time', validate=fields.validate.Length(max=26))
    last_modified = fields.Str(format='date-time', validate=fields.validate.Length(max=26))
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    version = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    ngc_path = fields.Str(format="regex", regex=r'^\w+(/[\w-]+)?/[\w-]+:[\w.-]+$', validate=fields.validate.Length(max=250))
    sha256_digest = fields.Dict(allow_none=True)
    is_ptm_backbone = fields.Bool()
    base_experiment_pull_complete = EnumField(PullStatus)
    additional_id_info = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(keys=fields.Str(format="regex", regex=r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$', validate=fields.validate.Length(max=100), allow_none=True), values=fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True))
    encryption_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    network_arch = EnumField(ExperimentNetworkArchEnum)
    base_experiment = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36)), validate=validate.Length(max=2))
    dataset_type = EnumField(DatasetTypeEnum)
    eval_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    inference_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    calibration_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    train_datasets = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36)), validate=validate.Length(max=sys.maxsize))
    read_only = fields.Bool()
    public = fields.Bool()
    actions = fields.List(EnumField(ActionEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    jobs = fields.List(fields.Nested(ExperimentJobSchema), validate=validate.Length(max=sys.maxsize))
    automl_enabled = fields.Bool(allow_none=True)
    automl_algorithm = EnumField(AutoMLAlgorithm, allow_none=True)
    automl_max_recommendations = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_delete_intermediate_ckpt = fields.Bool(allow_none=True)
    override_automl_disabled_params = fields.Bool(allow_none=True)
    automl_R = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_nu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    epoch_multiplier = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_add_hyperparameters = fields.Str(format="regex", regex=r'\[.*\]', validate=fields.validate.Length(max=5000), allow_none=True)
    automl_remove_hyperparameters = fields.Str(format="regex", regex=r'\[.*\]', validate=fields.validate.Length(max=5000), allow_none=True)
    type = EnumField(ExperimentTypeEnum, default=ExperimentTypeEnum.vision, allow_none=True)
    realtime_infer = fields.Bool(allow_none=True)
    realtime_infer_support = fields.Bool()
    realtime_infer_endpoint = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    realtime_infer_model_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    model_params = fields.Dict(allow_none=True)
    realtime_infer_request_timeout = fields.Int(format="int64", validate=validate.Range(min=0, max=86400), allow_none=True)
    bundle_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)


class ExperimentListRspSchema(Schema):
    """Class defining experiment list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    experiments = fields.List(fields.Nested(ExperimentRspSchema), validate=validate.Length(max=sys.maxsize))


class ExperimentJobListSchema(Schema):
    """Class defining experiment job list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    jobs = fields.List(fields.Nested(ExperimentJobSchema), validate=validate.Length(max=sys.maxsize))


class ExperimentDownloadSchema(Schema):
    """Class defining experiment artifacts download schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    export_type = EnumField(ExperimentExportTypeEnum)


@app.route('/api/v1/users/<user_id>/experiments', methods=['GET'])
@disk_space_check
def experiment_list(user_id):
    """List Experiments.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Experiments
      description: Returns the list of Experiments
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Optional name filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: type
        in: query
        description: Optional type filter
        required: false
        schema:
          type: string
          enum: ["vision", "medical"]
      - name: arch
        in: query
        description: Optional network architecture filter
        required: false
        schema:
          type: string
          enum: [ "detectnet_v2" ]
      - name: read_only
        in: query
        description: Optional read_only filter
        required: false
        allowEmptyValue: true
        schema:
          type: boolean
      - name: user_only
        in: query
        description: Optional filter to select user owned experiments only
        required: false
        schema:
          type: boolean
      responses:
        200:
          description: Returned the list of Experiments
          content:
            application/json:
              schema:
                type: array
                items: ExperimentRspSchema
                maxItems: 2147483647
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    user_only = str(request.args.get('user_only', None)) in {'True', 'yes', 'y', 'true', 't', '1', 'on'}
    experiments = app_handler.list_experiments(user_id, user_only)
    filtered_experiments = filtering.apply(request.args, experiments)
    paginated_experiments = pagination.apply(request.args, filtered_experiments)
    pagination_total = len(filtered_experiments)
    metadata = {"experiments": paginated_experiments}
    schema = ExperimentListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))['experiments']))
    response.headers['X-Pagination-Total'] = str(pagination_total)
    return response


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>', methods=['GET'])
@disk_space_check
def experiment_retrieve(user_id, experiment_id):
    """Retrieve Experiment.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Experiment
      description: Returns the Experiment
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID of Experiment to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.retrieve_experiment(user_id, experiment_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>', methods=['DELETE'])
@disk_space_check
def experiment_delete(user_id, experiment_id):
    """Delete Experiment.
    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete Experiment
      description: Cancels all related running jobs and returns the deleted Experiment
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID of Experiment to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the deleted Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_experiment(user_id, experiment_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments', methods=['POST'])
@disk_space_check
def experiment_create(user_id):
    """Create new Experiment.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Create new Experiment
      description: Returns the new Experiment
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentReqSchema
        description: Initial metadata for new Experiment (base_experiment or network_arch required)
        required: true
      responses:
        201:
          description: Returned the new Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.create_experiment(user_id, request_dict)
    # Get schema
    schema = None
    if response.code == 201:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>', methods=['PUT'])
@disk_space_check
def experiment_update(user_id, experiment_id):
    """Update Experiment.
    ---
    put:
      tags:
      - EXPERIMENT
      summary: Update Experiment
      description: Returns the updated Experiment
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID of Experiment to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentReqSchema
        description: Updated metadata for Experiment
        required: true
      responses:
        200:
          description: Returned the updated Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_experiment(user_id, experiment_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>', methods=['PATCH'])
@disk_space_check
def experiment_partial_update(user_id, experiment_id):
    """Partial update Experiment.
    ---
    patch:
      tags:
      - EXPERIMENT
      summary: Partial update Experiment
      description: Returns the updated Experiment
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID of Experiment to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentReqSchema
        description: Updated metadata for Experiment
        required: true
      responses:
        200:
          description: Returned the updated Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_experiment(user_id, experiment_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def specs_schema_without_handler_id(user_id, action):
    """Retrieve Specs schema.
    ---
    get:
      summary: Retrieve Specs schema without experiment or dataset id
      description: Returns the Specs schema for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [ "dataset_convert", "convert", "convert_and_index", "convert_efficientdet_tf1", "convert_efficientdet_tf2", "kmeans", "augment", "train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference", "annotation", "analyze", "validate", "generate", "calibration_tensorfile", "confmat" ]
      responses:
        200:
          description: Returned the Specs schema for given action and network
          content:
            application/json:
              schema:
                type: object
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    network = request.args.get('network')
    format = request.args.get('format')
    train_datasets = request.args.getlist('train_datasets')

    response = app_handler.get_spec_schema_without_handler_id(user_id, network, format, action, train_datasets)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def experiment_specs_schema(user_id, experiment_id, action):
    """Retrieve Specs schema.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Specs schema
      description: Returns the Specs schema for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [ "dataset_convert", "convert", "convert_and_index", "convert_efficientdet_tf1", "convert_efficientdet_tf2", "kmeans", "augment", "train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference", "annotation", "analyze", "validate", "generate", "calibration_tensorfile", "confmat" ]
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_spec_schema(user_id, experiment_id, action, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs', methods=['POST'])
@disk_space_check
def experiment_job_run(user_id, experiment_id):
    """Run Experiment Jobs.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Run Experiment Jobs
      description: Asynchronously starts a Experiment Action and returns corresponding Job ID
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentActions
      responses:
        201:
          description: Returned the Job ID corresponding to requested Experiment Action
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = ExperimentActions()
    request_schema_data = schema.dump(schema.load(request_data))
    requested_job = request_schema_data.get('parent_job_id', None)
    if requested_job:
        requested_job = str(requested_job)
    requested_action = request_schema_data.get('action', "")
    specs = request_schema_data.get('specs', {})
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    if isinstance(specs, dict) and "cluster" in specs:
        metadata = {"error_desc": "cluster is an invalid spec", "error_code": 3}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_run(user_id, experiment_id, requested_job, requested_action, "experiment", specs=specs, name=name, description=description)
    # Get schema
    schema = None
    if response.code == 201:
        if hasattr(response, "attachment_key") and response.attachment_key:
            send_file_response = send_file(response.data[response.attachment_key], as_attachment=True)
            # send_file sets correct response code as 200, should convert back to 201
            if send_file_response.status_code == 200:
                send_file_response.status_code = response.code
                # remove sent file as it's useless now
                os.remove(response.data[response.attachment_key])
            return send_file_response
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs', methods=['GET'])
@disk_space_check
def experiment_job_list(user_id, experiment_id):
    """List Jobs for Experiment.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Jobs for Experiment
      description: Returns the list of Jobs
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          pattern: '.*'
          maxLength: 36
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending" ]
      responses:
        200:
          description: Returned list of Jobs
          content:
            application/json:
              schema:
                type: array
                items: ExperimentJobSchema
                maxItems: 2147483647
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=None if experiment_id in ("*", "all") else experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_list(user_id, experiment_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        pagination_total = 0
        metadata = {"jobs": response.data}
        schema = ExperimentJobListSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))['jobs']))
        response.headers['X-Pagination-Total'] = str(pagination_total)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def experiment_job_retrieve(user_id, experiment_id, job_id):
    """Retrieve Job for Experiment.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Job for Experiment
      description: Returns the Job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job
          content:
            application/json:
              schema: ExperimentJobSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_retrieve(user_id, experiment_id, job_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentJobSchema()
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def experiment_job_cancel(user_id, experiment_id, job_id):
    """Cancel Experiment Job (or pause training).
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel Experiment Job or pause training
      description: Cancel Experiment Job or pause training
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested cancelation or training pause of specified Job ID (asynchronous)
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_cancel(user_id, experiment_id, job_id, "experiment")
    # Get schema
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def experiment_job_delete(user_id, experiment_id, job_id):
    """Delete Experiment Job.
    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete Experiment Job
      description: Delete Experiment Job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested deletion of specified Job ID
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_delete(user_id, experiment_id, job_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs/<job_id>:resume', methods=['POST'])
@disk_space_check
def experiment_job_resume(user_id, experiment_id, job_id):
    """Resume Experiment Job - train/retrain only.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Resume Experiment Job
      description: Resume Experiment Job - train/retrain only
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested resume of specified Job ID (asynchronous)
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = ExperimentActions()
    request_schema_data = schema.dump(schema.load(request_data))
    parent_job_id = request_schema_data.get('parent_job_id', None)
    if parent_job_id:
        parent_job_id = str(parent_job_id)
    specs = request_schema_data.get('specs', {})
    # Get response
    response = app_handler.resume_experiment_job(user_id, experiment_id, job_id, parent_job_id, "experiment", specs=specs)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def experiment_job_download(user_id, experiment_id, job_id):
    """Download Job Artifacts.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Download Job Artifacts
      description: Download the Artifacts produced by a given job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
                maxLength: 1000
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True, silent=True)
    request_data = {} if request_data is None else request_data
    try:
        request_schema_data = ExperimentDownloadSchema().load(request_data)
    except exceptions.ValidationError as err:
        metadata = {"error_desc": str(err)}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 404)
        return response
    export_type = request_schema_data.get("export_type", ExperimentExportTypeEnum.tao)
    # Get response
    response = app_handler.job_download(user_id, experiment_id, job_id, "experiment", export_type=export_type.name)
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def experiment_job_files_list(user_id, experiment_id, job_id):
    """List Job Files.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Job Files
      description: List the Files produced by a given job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Files
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
                  maxLength: 1000
                  maxLength: 1000
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    retrieve_logs = ast.literal_eval(request.args.get("retrieve_logs", "False"))
    retrieve_specs = ast.literal_eval(request.args.get("retrieve_specs", "False"))
    response = app_handler.job_list_files(user_id, experiment_id, job_id, retrieve_logs, retrieve_specs, "experiment")
    # Get schema
    if response.code == 200:
        if isinstance(response.data, list) and (all(isinstance(f, str) for f in response.data) or response.data == []):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: file list invalid", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/users/<user_id>/experiments/<experiment_id>/jobs/<job_id>:download_selective_files', methods=['GET'])
@disk_space_check
def experiment_job_download_selective_files(user_id, experiment_id, job_id):
    """Download selective Job Artifacts.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Download selective Job Artifacts
      description: Download selective Artifacts produced by a given job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(user_id=user_id, experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    file_lists = request.args.getlist('file_lists')
    best_model = ast.literal_eval(request.args.get('best_model', "False"))
    latest_model = ast.literal_eval(request.args.get('latest_model', "False"))
    tar_files = ast.literal_eval(request.args.get('tar_files', "True"))
    if not (file_lists or best_model or latest_model):
        return make_response(jsonify("No files passed in list format to download or, best_model or latest_model is not enabled"), 400)
    # Get response
    response = app_handler.job_download(user_id, experiment_id, job_id, "experiment", file_lists=file_lists, best_model=best_model, latest_model=latest_model, tar_files=tar_files)
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


#
# HEALTH API
#
@app.route('/api/v1/health', methods=['GET'])
def api_health():
    """api health endpoint"""
    return make_response(jsonify(['liveness', 'readiness']))


@app.route('/api/v1/health/liveness', methods=['GET'])
@disk_space_check
def liveness():
    """api liveness endpoint"""
    live_state = health_check.check_logging()
    if live_state:
        return make_response(jsonify("OK"), 201)
    return make_response(jsonify("Error"), 400)


@app.route('/api/v1/health/readiness', methods=['GET'])
@disk_space_check
def readiness():
    """api readiness endpoint"""
    ready_state = health_check.check_logging() and health_check.check_k8s() and Workflow.healthy()
    if ready_state:
        return make_response(jsonify("OK"), 201)
    return make_response(jsonify("Error"), 400)


#
# BASIC API
#
@app.route('/', methods=['GET'])
@disk_space_check
def root():
    """api root endpoint"""
    return make_response(jsonify(['api', 'openapi.yaml', 'openapi.json', 'redoc', 'swagger', 'tao_api_notebooks.zip']))


@app.route('/api', methods=['GET'])
def version_list():
    """version list endpoint"""
    return make_response(jsonify(['v1']))


@app.route('/api/v1', methods=['GET'])
def version_v1():
    """version endpoint"""
    return make_response(jsonify(['login', 'user', 'auth', 'health']))


@app.route('/api/v1/users', methods=['GET'])
def user_list():
    """user list endpoint"""
    error = {"error_desc": "Listing users is not authorized: Missing User ID", "error_code": 1}
    schema = ErrorRspSchema()
    return make_response(jsonify(schema.dump(schema.load(error))), 403)


@app.route('/api/v1/users/<user_id>', methods=['GET'])
@disk_space_check
def user(user_id):
    """user endpoint"""
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    return make_response(jsonify(['dataset', 'experiment']))


@app.route('/openapi.yaml', methods=['GET'])
def openapi_yaml():
    """openapi_yaml endpoint"""
    r = make_response(spec.to_yaml())
    r.mimetype = 'text/x-yaml'
    return r


@app.route('/openapi.json', methods=['GET'])
def openapi_json():
    """openapi_json endpoint"""
    r = make_response(jsonify(spec.to_dict()))
    r.mimetype = 'application/json'
    return r


@app.route('/redoc', methods=['GET'])
def redoc():
    """redoc endpoint"""
    return render_template('redoc.html')


@app.route('/swagger', methods=['GET'])
def swagger():
    """swagger endpoint"""
    return render_template('swagger.html')


@app.route('/tao_api_notebooks.zip', methods=['GET'])
@disk_space_check
def download_folder():
    """Download notebooks endpoint"""
    # Create a temporary zip file containing the folder
    shutil.make_archive("/tmp/tao_api_notebooks", 'zip', "/shared/notebooks/")

    # Send the zip file for download
    return send_file(
        "/tmp/tao_api_notebooks.zip",
        as_attachment=True,
        download_name="tao_api_notebooks.zip"
    )


#
# End of APIs
#


#
# Cache part
#

time_loop = Timeloop()


@time_loop.job(interval=timedelta(seconds=300))
def clear_cache():
    """Clear cache every 5 minutes"""
    from handlers.medical_dataset_handler import MedicalDatasetHandler
    MedicalDatasetHandler.clean_cache()


@atexit.register
def stop_clear_cache():
    """Stop cache clear thread"""
    time_loop.stop()
    print("Exit cache clear thread.")


#
# End cache part
#

with app.test_request_context():
    spec.path(view=login)
    spec.path(view=dataset_list)
    spec.path(view=dataset_retrieve)
    spec.path(view=dataset_delete)
    spec.path(view=dataset_create)
    spec.path(view=dataset_update)
    spec.path(view=dataset_partial_update)
    spec.path(view=dataset_upload)
    spec.path(view=dataset_specs_schema)
    spec.path(view=dataset_job_run)
    spec.path(view=dataset_job_list)
    spec.path(view=dataset_job_retrieve)
    spec.path(view=dataset_job_cancel)
    spec.path(view=dataset_job_delete)
    spec.path(view=dataset_job_download)
    spec.path(view=experiment_list)
    spec.path(view=experiment_retrieve)
    spec.path(view=experiment_delete)
    spec.path(view=experiment_create)
    spec.path(view=experiment_update)
    spec.path(view=experiment_partial_update)
    spec.path(view=experiment_specs_schema)
    spec.path(view=experiment_job_run)
    spec.path(view=experiment_job_list)
    spec.path(view=experiment_job_retrieve)
    spec.path(view=experiment_job_cancel)
    spec.path(view=experiment_job_delete)
    spec.path(view=experiment_job_resume)
    spec.path(view=experiment_job_download)


@synchronized
def scan_for_workspaces():
    """Scan to sync workspaces mounting points"""
    while True:
        report_healthy("App has waken up.")

        if not os.path.exists("/shared/"):
            continue

        # Sync workspaces
        for user_id in os.listdir("/shared/users"):
            if os.path.isfile(f"/shared/users/{user_id}"):
                continue
            workspace_metadata_file, workspaces = load_user_workspace_metadata(user_id)
            workspace_metadata_updated = False

            for handler_id in workspaces.keys():
                workspace = workspaces[handler_id]
                if workspace["status"] == "DELETED":
                    continue

                user_id = workspace["user_id"]
                mount_path = workspace["mount_path"]

                if not workspace_info(user_id, handler_id):
                    unmount_ngc_workspace(None, user_id, None, handler_id)
                    if os.path.exists(mount_path):
                        deletion_command = f"rm -rf {mount_path}"
                        delete_thread = threading.Thread(target=run_system_command, args=(deletion_command,))
                        delete_thread.start()
                        print("Workspace local path deleted successfully.", file=sys.stderr)
                    # Update status to DELETED
                    workspace["status"] = "DELETED"
                    workspace_metadata_updated = True
                elif not os.path.ismount(mount_path):
                    # Mount workspaces
                    mount_ngc_workspace(user_id, handler_id)

            if workspace_metadata_updated:
                safe_dump_file(workspace_metadata_file, workspaces)

        report_healthy("App daemon is going to sleep.")
        time.sleep(15)


def AppThreadStart():
    """Initialize app daemon. Starts a thread if thread not exists"""
    for thread in threading.enumerate():
        if thread.name == "AppThreadTAO":
            return False
    t = threading.Thread(target=scan_for_workspaces)
    t.name = 'AppThreadTAO'
    t.daemon = True
    t.start()
    return True


if os.getenv("DEV_MODE", "False").lower() not in ("true", "1"):
    import uwsgi
    if os.getenv("NGC_RUNNER", "") == "True" and uwsgi.worker_id() == 1:
        """Start APP Daemon for ngc workspace monitor"""
        AppThreadStart()
elif os.getenv("NGC_RUNNER", "") == "True":
    AppThreadStart()


if __name__ == '__main__':
    time_loop.start()

    # app.run(host='0.0.0.0', port=8000)
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        app.run(host="0.0.0.0", port=8008)
    else:
        app.run()
