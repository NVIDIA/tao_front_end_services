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
import re

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from flask import Flask, request, jsonify, make_response, render_template, send_from_directory, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from requests_toolbelt.multipart.encoder import MultipartEncoder
from marshmallow import Schema, fields, exceptions, validate, validates_schema, ValidationError, EXCLUDE
from marshmallow_enum import EnumField, Enum

from filter_utils import filtering, pagination
from auth_utils import credentials, authentication, access_control, metrics
from health_utils import health_check

from enum_constants import DatasetFormat, DatasetType, ExperimentNetworkArch, Metrics
from handlers.app_handler import AppHandler as app_handler
from handlers.stateless_handlers import resolve_metadata, get_root, get_metrics, set_metrics
from handlers.utilities import validate_uuid
from utils import is_pvc_space_free, safe_load_file, log_monitor, log_api_error, is_cookie_request, DataMonitorLogTypeEnum
from job_utils.workflow import Workflow

from werkzeug.exceptions import HTTPException
from werkzeug.middleware.profiler import ProfilerMiddleware
from datetime import timedelta, datetime
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
        threshold_bytes = 100 * 1024 * 1024

        pvc_free_space, pvc_free_bytes = is_pvc_space_free(threshold_bytes)
        msg = f"PVC free space remaining is {pvc_free_bytes} bytes which is less than {threshold_bytes} bytes"
        if not pvc_free_space:
            return make_response(jsonify({'error': f'Disk space is nearly full. {msg}. Delete appropriate experiments/datasets'}), 500)

        return f(*args, **kwargs)

    return decorated_function


#
# Create an APISpec
#
tao_version = os.environ.get('TAO_VERSION', 'unknown')
spec = APISpec(
    title='NVIDIA TAO API',
    version=tao_version,
    openapi_version='3.0.3',
    info={"description": 'NVIDIA TAO (Train, Adapt, Optimize) API document'},
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
spec.components.header("Access-Control-Allow-Origin", {
    "description": "Origins that are allowed to share response",
    "schema": {
        "type": "string",
        "format": "regex",
        "maxLength": sys.maxsize,
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


class EnumFieldPrefix(fields.Field):
    """Enum field override for Metrics"""

    def __init__(self, enum, *args, **kwargs):
        """Init function of class"""
        self.enum = enum
        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        if value in self.enum._value2member_map_:
            return value
        # Check for best_ prefixed values
        if value.startswith('best_'):
            base_value = value[5:]
            if base_value in self.enum._value2member_map_:
                return value
        raise ValidationError(f"Invalid value '{value}' for enum '{self.enum.__name__}'")

    def _serialize(self, value, attr, obj, **kwargs):
        return value


marshmallow_plugin.converter.add_attribute_function(enum_to_properties)


#
# Global schemas and enums
#
class MessageOnlySchema(Schema):
    """Class defining dataset upload schema"""

    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))


class ErrorRspSchema(Schema):
    """Class defining error response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    error_desc = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    error_code = fields.Int(validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize), format=sys_int_format())


class JobStatusEnum(Enum):
    """Class defining job status enum"""

    Done = 'Done'
    Running = 'Running'
    Error = 'Error'
    Pending = 'Pending'
    Canceled = 'Canceled'
    Canceling = 'Canceling'
    Pausing = 'Pausing'
    Paused = 'Paused'
    Resuming = 'Resuming'


class JobPlatformEnum(Enum):
    """Class defining job platform enum"""

    t4 = 't4'
    l4 = 'l4'
    l40 = 'l40'
    a10 = 'a10'
    a30 = 'a30'
    a40 = 'a40'
    a100 = 'a100'
    v100 = 'v100'


class PullStatus(Enum):
    """Class defining artifact upload/download status"""

    starting = "starting"
    in_progress = "in_progress"
    pull_complete = "pull_complete"
    invalid_pull = "invalid_pull"


class PaginationInfoSchema(Schema):
    """Class defining pagination info schema"""

    total_records = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    total_pages = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    page_size = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    page_index = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())


#
# Flask app
#

class CustomProfilerMiddleware(ProfilerMiddleware):
    """Class defining custom middleware to exclude health related endpoints from profiling"""

    def __call__(self, environ, start_response):
        """Wrapper around ProfilerMiddleware to only perform profiling for non health related API requests"""
        if '/api/v1/health' in environ['PATH_INFO']:
            return self._app(environ, start_response)
        return super().__call__(environ, start_response)


app = Flask(__name__)
app.json.sort_keys = False
app.config['TRAP_HTTP_EXCEPTIONS'] = True
if os.getenv("PROFILER", "FALSE") == "True":
    app.config["PROFILE"] = True
    app.wsgi_app = CustomProfilerMiddleware(
        app.wsgi_app,
        stream=sys.stderr,
        sort_by=('cumtime',),
        restrictions=[50],
    )
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


@app.errorhandler(exceptions.ValidationError)
def handle_validation_exception(e):
    """Return 400 bad request for validation exceptions"""
    metadata = {"error_desc": str(e)}
    schema = ErrorRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    return response


#
# JobResultSchema
#
class DetailedStatusSchema(Schema):
    """Class defining Status schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    date = fields.Str(format="mm/dd/yyyy", validate=fields.validate.Length(max=26))
    time = fields.Str(format="hh:mm:ss", validate=fields.validate.Length(max=26))
    message = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=6400))
    status = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class GraphSchema(Schema):
    """Class defining Graph schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    x_min = fields.Int(allow_none=True, validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize), format=sys_int_format())
    x_max = fields.Int(allow_none=True, validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize), format=sys_int_format())
    y_min = fields.Float(allow_none=True)
    y_max = fields.Float(allow_none=True)
    values = fields.Dict(keys=fields.Str(allow_none=True), values=fields.Float(allow_none=True))
    units = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class CategoryWiseSchema(Schema):
    """Class defining CategoryWise schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    category = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    value = fields.Float(allow_none=True)


class CategorySchema(Schema):
    """Class defining Category schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    category_wise_values = fields.List(fields.Nested(CategoryWiseSchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))


class KPISchema(Schema):
    """Class defining KPI schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    values = fields.Dict(allow_none=True)


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
    metric = EnumFieldPrefix(Metrics)
    value = CustomFloatField(allow_none=True)


class AutoMLResultsDetailedSchema(Schema):
    """Class defining AutoML detailed results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    current_experiment_id = fields.Int(allow_none=True, validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    best_experiment_id = fields.Int(allow_none=True, validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    metric = EnumFieldPrefix(Metrics)
    experiments = fields.Raw()


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
        unknown = EXCLUDE
    detailed_status = fields.Nested(DetailedStatusSchema, allow_none=True)
    graphical = fields.List(fields.Nested(GraphSchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    categorical = fields.List(fields.Nested(CategorySchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    kpi = fields.List(fields.Nested(KPISchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    automl_result = fields.List(fields.Nested(AutoMLResultsSchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    stats = fields.List(fields.Nested(StatsSchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    starting_epoch = fields.Int(allow_none=True, validate=fields.validate.Range(min=-1, max=sys.maxsize), format=sys_int_format(), error="Epoch should be larger than -1. With -1 meaning non-valid.")  # Epoch where kpi infos are included first
    epoch = fields.Int(allow_none=True, validate=fields.validate.Range(min=-1, max=sys.maxsize), format=sys_int_format(), error="Epoch should be larger than -1. With -1 meaning non-valid.")
    automl_experiment_epoch = fields.Int(allow_none=True, validate=fields.validate.Range(min=-1, max=sys.maxsize), format=sys_int_format(), error="Epoch should be larger than -1. With -1 meaning non-valid.")
    max_epoch = fields.Int(allow_none=True, validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format(), error="Max epoch should be non negative.")
    automl_experiment_max_epoch = fields.Int(allow_none=True, validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format(), error="Max epoch should be non negative.")
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
        unknown = EXCLUDE
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        401:
          description: Unauthorized
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
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
        if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'basic':
            basic_auth = request.authorization
            if basic_auth:
                if basic_auth.username == '$oauthtoken':
                    key = basic_auth.password
                    creds, err = credentials.get_from_ngc(key)
                    if 'token' in creds:
                        token = creds['token']
                # special metrics case
                elif basic_auth.username == '$metricstoken' and url.split('/', 3)[-1] == 'api/v1/metrics':
                    key = basic_auth.password
                    if metrics.validate(key):
                        return make_response(jsonify({}), 200)
                    metadata = {"error_desc": "wrong metrics key", "error_code": 1}
                    schema = ErrorRspSchema()
                    response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
                    return response
    sid_cookie = request.cookies.get('SID')
    ssid_cookie = request.cookies.get('SSID')
    if not token:
        if sid_cookie:
            token = 'SID=' + sid_cookie
    if not token:
        if ssid_cookie:
            token = 'SSID=' + ssid_cookie
    print('Token: ...' + str(token)[-10:], flush=True)
    # authentication
    user_id, org_name, err = authentication.validate(url, token)
    from_ui = is_cookie_request(request)
    log_content = f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, method:{method}, url:{url}"
    log_monitor(log_type=DataMonitorLogTypeEnum.api, log_content=log_content)
    credentials.save_cookie(user_id, sid_cookie, ssid_cookie)
    if err:
        print("Unauthorized: " + str(err), flush=True)
        metadata = {"error_desc": str(err), "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
        return response
    # access control
    err = access_control.validate(user_id, org_name, url)
    if err:
        print("Forbidden: " + str(err), flush=True)
        metadata = {"error_desc": str(err), "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 403)
        return response
    return make_response(jsonify({'user_id': user_id}), 200)


#
# Metrics API
#

class TelemetryReqSchema(Schema):
    """Class defining telemetry request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    version = fields.Str()
    network = fields.Str()
    action = fields.Str()
    success = fields.Bool()
    gpu = fields.List(fields.Str())
    time_lapsed = fields.Int()


@app.route('/api/v1/metrics', methods=['POST'])
def metrics_upsert():
    """Report execution of new action.
    ---
    post:
        tags:
        - TELEMETRY
        summary: Report execution of new action
        description: Post anonymous metrics to NVIDIA Kratos
        requestBody:
            content:
                application/json:
                    schema: TelemetryReqSchema
                    description: Report new action, network and gpu list
                    required: true
        responses:
            201:
                description: Sucessfully reported execution of new action
            400:
                description: Bad request, see reply body for details
                content:
                    application/json:
                        schema: ErrorRspSchema
    """
    now = old_now = datetime.now()

    # get action report

    try:
        data = TelemetryReqSchema().load(request.get_json(force=True))
    except:
        return make_response(jsonify({}), 400)

    # update metrics.json

    metrics = get_metrics()
    if not metrics:
        metrics = safe_load_file(os.path.join(get_root(), 'metrics.json'))
        if not metrics:
            metadata = {"error_desc": "Metrics.json file not exists or can not be updated now, please try again later.", "error_code": 503}
            schema = ErrorRspSchema()
            response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
            return response

    old_now = datetime.fromisoformat(metrics.get('last_updated', now.isoformat()))
    version = re.sub("[^a-zA-Z0-9]", "_", data.get('version', 'unknown')).lower()
    action = re.sub("[^a-zA-Z0-9]", "_", data.get('action', 'unknown')).lower()
    network = re.sub("[^a-zA-Z0-9]", "_", data.get('network', 'unknown')).lower()
    success = data.get('success', False)
    time_lapsed = data.get('time_lapsed', 0)
    gpus = data.get('gpu', ['unknown'])
    if success:
        metrics[f'total_action_{action}_pass'] = metrics.get(f'total_action_{action}_pass', 0) + 1
    else:
        metrics[f'total_action_{action}_fail'] = metrics.get(f'total_action_{action}_fail', 0) + 1
    metrics[f'version_{version}_action_{action}'] = metrics.get(f'version_{version}_action_{action}', 0) + 1
    metrics[f'network_{network}_action_{action}'] = metrics.get(f'network_{network}_action_{action}', 0) + 1
    metrics['time_lapsed_today'] = metrics.get('time_lapsed_today', 0) + time_lapsed
    if now.strftime("%d") != old_now.strftime("%d"):
        metrics['time_lapsed_today'] = time_lapsed
    for gpu in gpus:
        gpu = re.sub("[^a-zA-Z0-9]", "_", gpu).lower()
        metrics[f'gpu_{gpu}_action_{action}'] = metrics.get(f'gpu_{gpu}_action_{action}', 0) + 1
    metrics['last_updated'] = now.isoformat()

    set_metrics(metrics)

    # success

    return make_response(jsonify(metrics), 201)


# Define enum common to Dataset and Experiment Api

class ActionEnum(Enum):
    """Class defining action type enum"""

    dataset_convert = 'dataset_convert'
    convert = 'convert'
    convert_efficientdet_tf2 = 'convert_efficientdet_tf2'

    train = 'train'
    evaluate = 'evaluate'
    prune = 'prune'
    retrain = 'retrain'
    export = 'export'
    gen_trt_engine = 'gen_trt_engine'
    trtexec = 'trtexec'
    inference = 'inference'
    batchinfer = 'batchinfer'

    augment = 'augment'
    annotation_format_convert = 'annotation_format_convert'
    analyze = 'analyze'
    validate = 'validate'
    generate = 'generate'

    calibration_tensorfile = 'calibration_tensorfile'

    annotation = 'annotation'
    nextimage = 'nextimage'
    cacheimage = 'cacheimage'
    notify = 'notify'
    auto3dseg = 'auto3dseg'

#
# WORKSPACE API
#


class CloudPullTypesEnum(Enum):
    """Class defining cloud pull types enum"""

    aws = 'aws'
    azure = 'azure'
    gcp = 'gcp'
    huggingface = 'huggingface'
    self_hosted = 'self_hosted'


class AWSCloudPullSchema(Schema):
    """Class defining AWS Cloud pull schema"""

    access_key = fields.Str(validate=validate.Length(max=2048))
    secret_key = fields.Str(validate=validate.Length(max=2048))
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(max=2048), allow_none=True)


class AzureCloudPullSchema(Schema):
    """Class defining AWS Cloud pull schema"""

    account_name = fields.Str(validate=validate.Length(max=2048))
    access_key = fields.Str(validate=validate.Length(max=2048))
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(max=2048), allow_none=True)


class HuggingFaceCloudPullSchema(Schema):
    """Class defining Hugging Face Cloud pull schema"""

    token = fields.Str(validate=validate.Length(max=2048))


class CloudFileType(Enum):
    """Class defining cloud file types enum"""

    file = "file"
    folder = "folder"


class WorkspaceReqSchema(Schema):
    """Class defining Cloud Workspace request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    cloud_type = EnumField(CloudPullTypesEnum, allow_none=False)
    cloud_specific_details = fields.Field(allow_none=False)

    @validates_schema
    def validate_cloud_specific_details(self, data, **kwargs):
        """Return schema based on cloud_type"""
        cloud_type = data.get('cloud_type')

        if cloud_type:
            if cloud_type == CloudPullTypesEnum.aws:
                schema = AWSCloudPullSchema()
            elif cloud_type == CloudPullTypesEnum.azure:
                schema = AzureCloudPullSchema()
            elif cloud_type == CloudPullTypesEnum.huggingface:
                schema = HuggingFaceCloudPullSchema()
            else:
                schema = Schema()

            try:
                schema.load(data['cloud_specific_details'], unknown=EXCLUDE)
            except Exception as e:
                raise fields.ValidationError(str(e))


class DateTimeField(fields.DateTime):
    """Class defining datetime object deserialization (since marshmallow doesn't handle python date objects natively, expects a date string instead)"""

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, datetime):
            return value
        return super()._deserialize(value, attr, data, **kwargs)


class WorkspaceRspSchema(Schema):
    """Class defining Cloud pull schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField()
    last_modified = DateTimeField()
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    cloud_type = EnumField(CloudPullTypesEnum, allow_none=False)


class WorkspaceListRspSchema(Schema):
    """Class defining workspace list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    workspaces = fields.List(fields.Nested(WorkspaceRspSchema), validate=validate.Length(max=sys.maxsize))


class DatasetPathLstSchema(Schema):
    """Class defining dataset actions schema"""

    dataset_paths = fields.List(fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True), validate=validate.Length(max=sys.maxsize))


@app.route('/api/v1/orgs/<org_name>/workspaces', methods=['GET'])
@disk_space_check
def workspace_list(org_name):
    """List workspaces.
    ---
    get:
      tags:
      - WORKSPACE
      summary: List workspaces
      description: Returns the list of workspaces
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
          enum: ["monai", "unet", "custom" ]
      - name: type
        in: query
        description: Optional type filter
        required: false
        schema:
          type: string
          enum: [ "object_detection", "semantic_segmentation", "image_classification" ]
      responses:
        200:
          description: Returned list of workspaces
          content:
            application/json:
              schema:
                type: array
                items: WorkspaceRspSchema
                maxItems: 2147483647
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)
    workspaces = app_handler.list_workspaces(user_id, org_name)
    filtered_workspaces = filtering.apply(request.args, workspaces)
    paginated_workspaces = pagination.apply(request.args, filtered_workspaces)
    metadata = {"workspaces": paginated_workspaces}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_workspaces),
            "total_pages": math.ceil(len(filtered_workspaces) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = WorkspaceListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>', methods=['GET'])
@disk_space_check
def workspace_retrieve(org_name, workspace_id):
    """Retrieve Workspace.
    ---
    get:
      tags:
      - WORKSPACE
      summary: Retrieve Workspace
      description: Returns the Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.retrieve_workspace(org_name, workspace_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>:get_datasets', methods=['GET'])
@disk_space_check
def workspace_retrieve_datasets(org_name, workspace_id):
    """Retrieve Datasets from Workspace.
    ---
    get:
      tags:
      - WORKSPACE
      summary: Retrieve datasets from Workspace
      description: Returns the datasets matched with the request body args inside the Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned list of dataset paths within Workspace
          content:
            application/json:
              schema: DatasetPathLstSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    dataset_type = request.args.get("dataset_type", None)
    dataset_format = request.args.get("dataset_format", None)
    dataset_intention = request.args.getlist("dataset_intention")
    # Get response
    response = app_handler.retrieve_cloud_datasets(org_name, workspace_id, dataset_type, dataset_format, dataset_intention)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetPathLstSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>', methods=['DELETE'])
@disk_space_check
def workspace_delete(org_name, workspace_id):
    """Delete Workspace.
    ---
    delete:
      tags:
      - WORKSPACE
      summary: Delete Workspace
      description: Cancels all related running jobs and returns the deleted Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Deleted Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_workspace(org_name, workspace_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces', methods=['POST'])
@disk_space_check
def workspace_create(org_name):
    """Create new Workspace.
    ---
    post:
      tags:
      - WORKSPACE
      summary: Create new Workspace
      description: Returns the new Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema: WorkspaceReqSchema
        description: Initial metadata for new Workspace (type and format required)
        required: true
      responses:
        201:
          description: Retuned the new Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = WorkspaceReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)
    # Get response
    response = app_handler.create_workspace(user_id, org_name, request_dict)
    # Get schema
    schema = None
    if response.code == 201:
        schema = WorkspaceRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>', methods=['PUT'])
@disk_space_check
def workspace_update(org_name, workspace_id):
    """Update Workspace.
    ---
    put:
      tags:
      - WORKSPACE
      summary: Update Workspace
      description: Returns the updated Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: WorkspaceReqSchema
        description: Updated metadata for Workspace
        required: true
      responses:
        200:
          description: Returned the updated Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = WorkspaceReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_workspace(org_name, workspace_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>', methods=['PATCH'])
@disk_space_check
def workspace_partial_update(org_name, workspace_id):
    """Partial update Workspace.
    ---
    patch:
      tags:
      - WORKSPACE
      summary: Partial update Workspace
      description: Returns the updated Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: WorkspaceReqSchema
        description: Updated metadata for Workspace
        required: true
      responses:
        200:
          description: Returned the updated Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = WorkspaceRspSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_workspace(org_name, workspace_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)

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
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform = EnumField(JobPlatformEnum, allow_none=True)


class DatasetIntentEnum(Enum):
    """Class defining dataset intent enum"""

    training = 'training'
    evaluation = 'evaluation'
    testing = 'testing'


class LstStrSchema(Schema):
    """Class defining dataset actions schema"""

    dataset_formats = fields.List(fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True), validate=validate.Length(max=sys.maxsize))
    accepted_dataset_intents = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=3))


class DatasetReqSchema(Schema):
    """Class defining dataset request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    logo = fields.URL(validate=fields.validate.Length(max=2048))
    type = EnumField(DatasetType)
    format = EnumField(DatasetFormat)
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    url = fields.URL(validate=fields.validate.Length(max=2048))  # For HuggingFace and Self_hosted
    cloud_file_path = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)
    use_for = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=3))


class DatasetJobSchema(Schema):
    """Class defining dataset job result total schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    created_on = DateTimeField()
    last_modified = DateTimeField()
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    result = fields.Nested(JobResultSchema)
    specs = fields.Raw(allow_none=True)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform = EnumField(JobPlatformEnum, allow_none=True)
    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class DatasetRspSchema(Schema):
    """Class defining dataset response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        load_only = ("user_id", "docker_env_vars", "client_id", "client_secret", "filters")
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField()
    last_modified = DateTimeField()
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    type = EnumField(DatasetType)
    format = EnumField(DatasetFormat)
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    url = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)  # For HuggingFace and Self_hosted
    cloud_file_path = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    actions = fields.List(EnumField(ActionEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    jobs = fields.Dict(keys=fields.Str(format="uuid", validate=fields.validate.Length(max=36)), values=fields.Nested(DatasetJobSchema), validate=validate.Length(max=sys.maxsize))
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)
    use_for = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=3))


class DatasetListRspSchema(Schema):
    """Class defining dataset list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    datasets = fields.List(fields.Nested(DatasetRspSchema), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfoSchema, allowed_none=True)


class DatasetJobListSchema(Schema):
    """Class defining dataset list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    jobs = fields.List(fields.Nested(DatasetJobSchema), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfoSchema, allowed_none=True)


@app.route('/api/v1/orgs/<org_name>/datasets:get_formats', methods=['GET'])
def get_dataset_formats(org_name):
    """Get dataset formats supported.
    ---
    post:
        tags:
        - DATASET
        summary: Given dataset type return dataset formats or return all formats
        description: Given dataset type return dataset formats or return all formats
        parameters:
        - name: org_name
          in: path
          description: Org Name
          required: true
          schema:
            type: string
            maxLength: 255
            pattern: '^[a-zA-Z0-9_-]+$'
        responses:
          200:
            description: Returns a list of dataset formats supported
            content:
              application/json:
                schema: LstStrSchema
            headers:
              Access-Control-Allow-Origin:
                $ref: '#/components/headers/Access-Control-Allow-Origin'
              X-RateLimit-Limit:
                $ref: '#/components/headers/X-RateLimit-Limit'
          404:
            description: Bad request, see reply body for details
            content:
              application/json:
                schema: ErrorRspSchema
            headers:
              Access-Control-Allow-Origin:
                $ref: '#/components/headers/Access-Control-Allow-Origin'
              X-RateLimit-Limit:
                $ref: '#/components/headers/X-RateLimit-Limit'
    """
    dataset_type = str(request.args.get('dataset_type', ''))
    # Get response
    response = app_handler.get_dataset_formats(dataset_type)
    # Get schema
    schema = None
    if response.code == 200:
        schema = LstStrSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets', methods=['GET'])
@disk_space_check
def dataset_list(org_name):
    """List Datasets.
    ---
    get:
      tags:
      - DATASET
      summary: List Datasets
      description: Returns the list of Datasets
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
          enum: ["kitti", "pascal_voc", "raw", "coco_raw", "unet", "coco", "lprnet", "train", "test", "default", "custom", "classification_pyt", "classification_tf2", "visual_changenet_segment", "visual_changenet_classify"]
      - name: type
        in: query
        description: Optional type filter
        required: false
        schema:
          type: string
          enum: [ "object_detection", "semantic_segmentation", "image_classification", "instance_segmentation", "character_recognition", "action_recognition", "pointpillars", "pose_classification", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "re_identification", "visual_changenet", "centerpose" ]
      responses:
        200:
          description: Returned list of Datasets
          content:
            application/json:
              schema: DatasetListRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)
    datasets = app_handler.list_datasets(user_id, org_name)
    filtered_datasets = filtering.apply(request.args, datasets)
    paginated_datasets = pagination.apply(request.args, filtered_datasets)
    metadata = {"datasets": paginated_datasets}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_datasets),
            "total_pages": math.ceil(len(filtered_datasets) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = DatasetListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>', methods=['GET'])
@disk_space_check
def dataset_retrieve(org_name, dataset_id):
    """Retrieve Dataset.
    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Dataset
      description: Returns the Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.retrieve_dataset(org_name, dataset_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>', methods=['DELETE'])
@disk_space_check
def dataset_delete(org_name, dataset_id):
    """Delete Dataset.
    ---
    delete:
      tags:
      - DATASET
      summary: Delete Dataset
      description: Cancels all related running jobs and returns the deleted Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_dataset(org_name, dataset_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets', methods=['POST'])
@disk_space_check
def dataset_create(org_name):
    """Create new Dataset.
    ---
    post:
      tags:
      - DATASET
      summary: Create new Dataset
      description: Returns the new Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema: DatasetReqSchema
        description: Initial metadata for new Dataset (type and format required)
        required: true
      responses:
        201:
          description: Returned the new Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)
    from_ui = is_cookie_request(request)
    # Get response
    response = app_handler.create_dataset(user_id, org_name, request_dict, from_ui=from_ui)
    # Get schema
    schema = None
    if response.code == 201:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 201:
        ds_format = request_dict.get("format", "")
        log_type = DataMonitorLogTypeEnum.medical_dataset if ds_format == "monai" else DataMonitorLogTypeEnum.tao_dataset
        log_api_error(user_id, org_name, from_ui, schema_dict, log_type, action="creation")

    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>', methods=['PUT'])
@disk_space_check
def dataset_update(org_name, dataset_id):
    """Update Dataset.
    ---
    put:
      tags:
      - DATASET
      summary: Update Dataset
      description: Returns the updated Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_dataset(org_name, dataset_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>', methods=['PATCH'])
@disk_space_check
def dataset_partial_update(org_name, dataset_id):
    """Partial update Dataset.
    ---
    patch:
      tags:
      - DATASET
      summary: Partial update Dataset
      description: Returns the updated Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_dataset(org_name, dataset_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def dataset_specs_schema(org_name, dataset_id, action):
    """Retrieve Specs schema.
    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Specs schema
      description: Returns the Specs schema for a given action
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
          enum: [ "dataset_convert", "convert", "convert_efficientdet_tf2", "kmeans", "augment", "train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference", "annotation", "analyze", "validate", "generate", "calibration_tensorfile" ]
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_spec_schema(org_name, dataset_id, action, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs', methods=['POST'])
@disk_space_check
def dataset_job_run(org_name, dataset_id):
    """Run Dataset Jobs.
    ---
    post:
      tags:
      - DATASET
      summary: Run Dataset Jobs
      description: Asynchronously starts a dataset action and returns corresponding Job ID
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
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
    num_gpu = request_schema_data.get('num_gpu', -1)
    platform = request_schema_data.get('platform', None)
    from_ui = is_cookie_request(request)
    # Get response
    response = app_handler.job_run(org_name, dataset_id, requested_job, requested_action, "dataset", specs=specs, name=name, description=description, num_gpu=num_gpu, platform=platform, from_ui=from_ui)
    handler_metadata = resolve_metadata("dataset", dataset_id)
    dataset_format = handler_metadata.get("format")
    # Get schema
    if response.code == 201:
        # MONAI dataset jobs are sync jobs and the response should be returned directly.
        if dataset_format == "monai":
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


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs', methods=['GET'])
@disk_space_check
def dataset_job_list(org_name, dataset_id):
    """List Jobs for Dataset.
    ---
    get:
      tags:
      - DATASET
      summary: List Jobs for Dataset
      description: Returns the list of Jobs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
              schema: DatasetJobListSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=None if dataset_id in ("*", "all") else dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)
    # Get response
    response = app_handler.job_list(user_id, org_name, dataset_id, "dataset")
    # Get schema
    if response.code == 200:
        filtered_jobs = filtering.apply(request.args, response.data)
        paginated_jobs = pagination.apply(request.args, filtered_jobs)
        metadata = {"jobs": paginated_jobs}
        # Pagination
        skip = request.args.get("skip", None)
        size = request.args.get("size", None)
        if skip is not None and size is not None:
            skip = int(skip)
            size = int(size)
            metadata["pagination_info"] = {
                "total_records": len(filtered_jobs),
                "total_pages": math.ceil(len(filtered_jobs) / size),
                "page_size": size,
                "page_index": skip // size,
            }
        schema = DatasetJobListSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))))
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def dataset_job_retrieve(org_name, dataset_id, job_id):
    """Retrieve Job for Dataset.
    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Job for Dataset
      description: Returns the Job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_retrieve(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetJobSchema()
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:status_update', methods=['POST'])
@disk_space_check
def dataset_job_status_update(org_name, dataset_id, job_id):
    """Update Job status for Dataset.
    ---
    get:
      tags:
      - DATASET
      summary: Posts status for the job
      description: Saves recieved content to status file
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = app_handler.job_status_update(org_name, dataset_id, job_id, "dataset", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:log_update', methods=['POST'])
@disk_space_check
def dataset_job_log_update(org_name, dataset_id, job_id):
    """Update Job log for Dataset.
    ---
    get:
      tags:
      - Dataset
      summary: Posts log for the job
      description: Saves recieved content to log file
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = app_handler.job_log_update(org_name, dataset_id, job_id, "dataset", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>/logs', methods=['GET'])
def dataset_job_logs(org_name, dataset_id, job_id):
    """Get realtime dataset job logs.
    ---
    get:
      tags:
      - DATASET
      summary: Get Job logs for Dataset
      description: Returns the job logs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
          description: Returned Job Logs
          content:
            text/plain:
              example: "Execution status: PASS"
          headers:
            Access-Control-Allow-Origin:
               $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Job not exist or logs not found.
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
               $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_job_logs(org_name, dataset_id, job_id, "dataset")
    if response.code == 200:
        response = make_response(response.data, 200)
        response.mimetype = 'text/plain'
        return response
    # Handle errors
    schema = ErrorRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(response.data))), 400)
    return response


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def dataset_job_cancel(org_name, dataset_id, job_id):
    """Cancel Dataset Job.
    ---
    post:
      tags:
      - DATASET
      summary: Cancel Dataset Job
      description: Cancel Dataset Job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_cancel(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def dataset_job_delete(org_name, dataset_id, job_id):
    """Delete Dataset Job.
    ---
    delete:
      tags:
      - DATASET
      summary: Delete Dataset Job
      description: delete Dataset Job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_delete(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def dataset_job_files_list(org_name, dataset_id, job_id):
    """List Job Files.
    ---
    get:
      tags:
      - DATASET
      summary: List Job Files
      description: List the Files produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    retrieve_logs = ast.literal_eval(request.args.get("retrieve_logs", "False"))
    retrieve_specs = ast.literal_eval(request.args.get("retrieve_specs", "False"))
    response = app_handler.job_list_files(org_name, dataset_id, job_id, retrieve_logs, retrieve_specs, "dataset")
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


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:download_selective_files', methods=['GET'])
@disk_space_check
def dataset_job_download_selective_files(org_name, dataset_id, job_id):
    """Download selective Job Artifacts.
    ---
    get:
      tags:
      - DATASET
      summary: Download selective Job Artifacts
      description: Download selective Artifacts produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
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
    response = app_handler.job_download(org_name, dataset_id, job_id, "dataset", file_lists=file_lists, tar_files=tar_files)
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


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def dataset_job_download(org_name, dataset_id, job_id):
    """Download Job Artifacts.
    ---
    get:
      tags:
      - DATASET
      summary: Download Job Artifacts
      description: Download the Artifacts produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_download(org_name, dataset_id, job_id, "dataset")
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
class LstIntSchema(Schema):
    """Class defining dataset actions schema"""

    data = fields.List(fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True), validate=validate.Length(max=sys.maxsize))


class ExperimentActions(Schema):
    """Class defining experiment actions schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform = EnumField(JobPlatformEnum, allow_none=True)


class PublishModel(Schema):
    """Class defining Publish model schema"""

    display_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    team_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    # format, framework, precision - to be determined by backend


class JobResumeSchema(Schema):
    """Class defining job resume request schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    specs = fields.Raw(allow_none=True)


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
    monai_bundle = 'monai_bundle'


class AutoMLAlgorithm(Enum):
    """Class defining automl algorithm enum"""

    bayesian = "bayesian"
    hyperband = "hyperband"


class AutoMLSchema(Schema):
    """Class defining automl parameters in a schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    automl_enabled = fields.Bool(allow_none=True)
    automl_algorithm = EnumField(AutoMLAlgorithm, allow_none=True)
    automl_max_recommendations = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_delete_intermediate_ckpt = fields.Bool(allow_none=True)
    override_automl_disabled_params = fields.Bool(allow_none=True)
    automl_R = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_nu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    epoch_multiplier = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_add_hyperparameters = fields.Str(format="regex", regex=r'\[.*\]', validate=fields.validate.Length(max=5000), allow_none=True)
    automl_remove_hyperparameters = fields.Str(format="regex", regex=r'\[.*\]', validate=fields.validate.Length(max=5000), allow_none=True)


class BaseExperimentMetadataSchema(Schema):
    """Class defining base experiment metadata schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    task = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    domain = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    backbone_type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    backbone_class = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_parameters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=10), allow_none=True)
    license = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    model_card_link = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    is_backbone = fields.Bool()
    is_trainable = fields.Bool()
    spec_file_present = fields.Bool()


class ExperimentReqSchema(Schema):
    """Class defining experiment request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))  # Model version description - not changing variable name for backward compatability
    model_description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))  # Description common to all versions of models
    version = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    logo = fields.URL(validate=fields.validate.Length(max=2048))
    ngc_path = fields.Str(format="regex", regex=r'^\w+(/[\w-]+)?/[\w-]+:[\w.-]+$', validate=fields.validate.Length(max=250))
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    sha256_digest = fields.Dict(allow_none=True)
    base_experiment_pull_complete = EnumField(PullStatus)
    additional_id_info = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(keys=fields.Str(format="regex", regex=r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$', validate=fields.validate.Length(max=100), allow_none=True), values=fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True))
    encryption_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    network_arch = EnumField(ExperimentNetworkArch)
    base_experiment = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36)), validate=validate.Length(max=2))
    eval_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    inference_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    calibration_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    train_datasets = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36)), validate=validate.Length(max=sys.maxsize))
    read_only = fields.Bool()
    public = fields.Bool()
    automl_settings = fields.Nested(AutoMLSchema, allow_none=True)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    type = EnumField(ExperimentTypeEnum, default=ExperimentTypeEnum.vision)
    realtime_infer = fields.Bool(default=False)
    model_params = fields.Dict(allow_none=True)
    bundle_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    realtime_infer_request_timeout = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    experiment_actions = fields.List(fields.Nested(ExperimentActions, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    tensorboard_enabled = fields.Bool(allow_none=True)
    tags = fields.List(fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500)), validate=validate.Length(max=sys.maxsize))
    retry_experiment_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class ExperimentJobSchema(Schema):
    """Class defining experiment job schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    created_on = DateTimeField()
    last_modified = DateTimeField()
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    result = fields.Nested(JobResultSchema)
    sync = fields.Bool()
    specs = fields.Raw(allow_none=True)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform = EnumField(JobPlatformEnum, allow_none=True)
    experiment_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class ExperimentRspSchema(Schema):
    """Class defining experiment response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        load_only = ("user_id", "docker_env_vars", "realtime_infer_endpoint", "realtime_infer_model_name")
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField()
    last_modified = DateTimeField()
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))  # Model version description - not changing variable name for backward compatability
    model_description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))  # Description common to all versions of models
    version = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    ngc_path = fields.Str(format="regex", regex=r'^\w+(/[\w-]+)?/[\w-]+:[\w.-]+$', validate=fields.validate.Length(max=250))
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    sha256_digest = fields.Dict(allow_none=True)
    base_experiment_pull_complete = EnumField(PullStatus)
    additional_id_info = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(keys=fields.Str(format="regex", regex=r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$', validate=fields.validate.Length(max=100), allow_none=True), values=fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True))
    encryption_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    network_arch = EnumField(ExperimentNetworkArch)
    base_experiment = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36)), validate=validate.Length(max=2))
    dataset_type = EnumField(DatasetType)
    dataset_formats = fields.List(EnumField(DatasetFormat), allow_none=True, validate=validate.Length(max=sys.maxsize))
    accepted_dataset_intents = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    eval_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    inference_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    calibration_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    train_datasets = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36)), validate=validate.Length(max=sys.maxsize))
    read_only = fields.Bool()
    public = fields.Bool()
    actions = fields.List(EnumField(ActionEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    jobs = fields.Dict(keys=fields.Str(format="uuid", validate=fields.validate.Length(max=36)), values=fields.Nested(ExperimentJobSchema), validate=validate.Length(max=sys.maxsize))
    all_jobs_cancel_status = EnumField(JobStatusEnum, allow_none=True)
    automl_settings = fields.Nested(AutoMLSchema)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    type = EnumField(ExperimentTypeEnum, default=ExperimentTypeEnum.vision, allow_none=True)
    realtime_infer = fields.Bool(allow_none=True)
    realtime_infer_support = fields.Bool()
    realtime_infer_endpoint = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    realtime_infer_model_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    model_params = fields.Dict(allow_none=True)
    realtime_infer_request_timeout = fields.Int(format="int64", validate=validate.Range(min=0, max=86400), allow_none=True)
    bundle_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    base_experiment_metadata = fields.Nested(BaseExperimentMetadataSchema, allow_none=True)
    experiment_actions = fields.List(fields.Nested(ExperimentActions, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    tensorboard_enabled = fields.Bool(default=False)
    tags = fields.List(fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500)), validate=validate.Length(max=sys.maxsize))


class ExperimentTagListSchema(Schema):
    """Class defining experiment tags list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    tags = fields.List(fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500)), validate=validate.Length(max=sys.maxsize))


class ExperimentListRspSchema(Schema):
    """Class defining experiment list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    experiments = fields.List(fields.Nested(ExperimentRspSchema), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfoSchema, allowed_none=True)


class ExperimentJobListSchema(Schema):
    """Class defining experiment job list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    jobs = fields.List(fields.Nested(ExperimentJobSchema), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfoSchema, allowed_none=True)


class ExperimentDownloadSchema(Schema):
    """Class defining experiment artifacts download schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    export_type = EnumField(ExperimentExportTypeEnum)


@app.route('/api/v1/orgs/<org_name>/experiments', methods=['GET'])
@disk_space_check
def experiment_list(org_name):
    """List Experiments.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Experiments
      description: Returns the list of Experiments
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
      - name: network_arch
        in: query
        description: Optional network architecture filter
        required: false
        schema:
          type: string
          enum: ["detectnet_v2", "unet", "classification_tf2", "efficientdet_tf2", "action_recognition", "classification_pyt", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "deformable_detr", "dino", "segformer", "visual_changenet", "centerpose"]
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
      - name: tag
        in: query
        description: Optional tag filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      responses:
        200:
          description: Returned the list of Experiments
          content:
            application/json:
              schema: ExperimentListRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_only = str(request.args.get('user_only', None)) in {'True', 'yes', 'y', 'true', 't', '1', 'on'}
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)
    experiments = app_handler.list_experiments(user_id, org_name, user_only)
    filtered_experiments = filtering.apply(request.args, experiments)
    paginated_experiments = pagination.apply(request.args, filtered_experiments)
    metadata = {"experiments": paginated_experiments}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_experiments),
            "total_pages": math.ceil(len(filtered_experiments) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = ExperimentListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/experiments:get_tags', methods=['GET'])
@disk_space_check
def experiment_tags_list(org_name):
    """Retrieve All Unique Experiment Tags.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve all unique experiment tags
      description: Returns all unique experiment tags
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      responses:
        200:
          description: Returned the unique experiment tags list
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)
    experiments = app_handler.list_experiments(user_id, org_name, user_only=True)
    unique_tags = list(set(tag for exp in experiments for tag in exp.get('tags', [])))
    metadata = {"tags": unique_tags}
    schema = ExperimentTagListSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/experiments:base', methods=['GET'])
@disk_space_check
def base_experiment_list(org_name):
    """List Base Experiments.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Experiments that can be used for transfer learning
      description: Returns the list of models published in NGC public catalog and private org's model registry
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
      - name: network_arch
        in: query
        description: Optional network architecture filter
        required: false
        schema:
          type: string
          enum: ["detectnet_v2", "unet", "classification_tf2", "efficientdet_tf2", "action_recognition", "classification_pyt", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "deformable_detr", "dino", "segformer", "visual_changenet", "centerpose"]
      - name: read_only
        in: query
        description: Optional read_only filter
        required: false
        allowEmptyValue: true
        schema:
          type: boolean
      responses:
        200:
          description: Returned the list of Experiments
          content:
            application/json:
              schema: ExperimentListRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    experiments = app_handler.list_base_experiments()
    filtered_experiments = filtering.apply(request.args, experiments)
    paginated_experiments = pagination.apply(request.args, filtered_experiments)
    metadata = {"experiments": paginated_experiments}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_experiments),
            "total_pages": math.ceil(len(filtered_experiments) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = ExperimentListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>', methods=['GET'])
@disk_space_check
def experiment_retrieve(org_name, experiment_id):
    """Retrieve Experiment.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Experiment
      description: Returns the Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.retrieve_experiment(org_name, experiment_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>', methods=['DELETE'])
@disk_space_check
def experiment_delete(org_name, experiment_id):
    """Delete Experiment.
    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete Experiment
      description: Cancels all related running jobs and returns the deleted Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_experiment(org_name, experiment_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments', methods=['POST'])
@disk_space_check
def experiment_create(org_name):
    """Create new Experiment.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Create new Experiment
      description: Returns the new Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)
    from_ui = is_cookie_request(request)
    # Get response
    response = app_handler.create_experiment(user_id, org_name, request_dict, from_ui=from_ui)
    # Get schema
    schema = None
    if response.code == 201:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 201:
        mdl_nw = request_dict.get("network_arch", None)
        is_medical = isinstance(mdl_nw, str) and mdl_nw.startswith("monai_")
        log_type = DataMonitorLogTypeEnum.medical_experiment if is_medical else DataMonitorLogTypeEnum.tao_experiment
        log_api_error(user_id, org_name, from_ui, schema_dict, log_type, action="creation")

    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>', methods=['PUT'])
@disk_space_check
def experiment_update(org_name, experiment_id):
    """Update Experiment.
    ---
    put:
      tags:
      - EXPERIMENT
      summary: Update Experiment
      description: Returns the updated Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_experiment(org_name, experiment_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>', methods=['PATCH'])
@disk_space_check
def experiment_partial_update(org_name, experiment_id):
    """Partial update Experiment.
    ---
    patch:
      tags:
      - EXPERIMENT
      summary: Partial update Experiment
      description: Returns the updated Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_experiment(org_name, experiment_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def specs_schema_without_handler_id(org_name, action):
    """Retrieve Specs schema.
    ---
    get:
      summary: Retrieve Specs schema without experiment or dataset id
      description: Returns the Specs schema for a given action
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [ "dataset_convert", "convert", "convert_efficientdet_tf2", "kmeans", "augment", "train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference", "annotation", "analyze", "validate", "generate", "calibration_tensorfile" ]
      responses:
        200:
          description: Returned the Specs schema for given action and network
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get response
    network = request.args.get('network')
    format = request.args.get('format')
    train_datasets = request.args.getlist('train_datasets')

    response = app_handler.get_spec_schema_without_handler_id(org_name, network, format, action, train_datasets)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def experiment_specs_schema(org_name, experiment_id, action):
    """Retrieve Specs schema.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Specs schema
      description: Returns the Specs schema for a given action
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
          enum: [ "dataset_convert", "convert", "convert_efficientdet_tf2", "kmeans", "augment", "train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference", "annotation", "analyze", "validate", "generate", "calibration_tensorfile" ]
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_spec_schema(org_name, experiment_id, action, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/specs/<action>/schema:base', methods=['GET'])
@disk_space_check
def base_experiment_specs_schema(org_name, experiment_id, action):
    """Retrieve Base Experiment Specs schema.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Base Experiment Specs schema
      description: Returns the Specs schema for a given action of the base experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Base Experiment
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
          enum: ["train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference", "generate" ]
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Action not found or Base spec file not present
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_base_experiment_spec_schema(experiment_id, action)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs', methods=['POST'])
@disk_space_check
def experiment_job_run(org_name, experiment_id):
    """Run Experiment Jobs.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Run Experiment Jobs
      description: Asynchronously starts a Experiment Action and returns corresponding Job ID
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
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
    requested_action = request_schema_data.get('action', None)
    if not requested_action:
        metadata = {"error_desc": "Action is required to run job", "error_code": 400}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    specs = request_schema_data.get('specs', {})
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    num_gpu = request_schema_data.get('num_gpu', -1)
    platform = request_schema_data.get('platform', None)
    from_ui = is_cookie_request(request)
    if isinstance(specs, dict) and "cluster" in specs:
        metadata = {"error_desc": "cluster is an invalid spec", "error_code": 3}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_run(org_name, experiment_id, requested_job, requested_action, "experiment", specs=specs, name=name, description=description, num_gpu=num_gpu, platform=platform, from_ui=from_ui)
    # Get schema
    schema = None
    if response.code == 201:
        if hasattr(response, "attachment_key") and response.attachment_key:
            try:
                output_path = response.data[response.attachment_key]
                all_files = [os.path.join(dirpath, f) for dirpath, dirnames, filenames in os.walk(output_path) for f in filenames]
                files_dict = {}
                for f in all_files:
                    with open(f, "rb") as file:
                        files_dict[os.path.relpath(f, output_path)] = file.read()
                multipart_data = MultipartEncoder(fields=files_dict)
                send_file_response = make_response(multipart_data.to_string())
                send_file_response.headers["Content-Type"] = multipart_data.content_type
                # send_file sets correct response code as 200, should convert back to 201
                if send_file_response.status_code == 200:
                    send_file_response.status_code = response.code
                    # remove sent file as it's useless now
                    shutil.rmtree(response.data[response.attachment_key], ignore_errors=True)
                return send_file_response
            except Exception as e:
                # get user_id for more information
                handler_metadata = resolve_metadata("experiment", experiment_id)
                user_id = handler_metadata.get("user_id")
                print(f"respond attached data for org: {org_name} experiment: {experiment_id} user: {user_id} failed, got error: {e}", file=sys.stderr)
                metadata = {"error_desc": "respond attached data failed", "error_code": 2}
                schema = ErrorRspSchema()
                response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
                return response
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 201:
        try:
            handler_metadata = resolve_metadata("experiment", experiment_id)
            is_medical = handler_metadata.get("type").lower() == "medical"
            user_id = handler_metadata.get("user_id", None)
            if user_id:
                log_type = DataMonitorLogTypeEnum.medical_job if is_medical else DataMonitorLogTypeEnum.tao_job
                log_api_error(user_id, org_name, from_ui, schema_dict, log_type, action="creation")
        except:
            log_monitor(DataMonitorLogTypeEnum.api, "Cannot parse experiment info for job.")

    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:publish_model', methods=['POST'])
@disk_space_check
def experiment_model_publish(org_name, experiment_id, job_id):
    """Publish models to NGC.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Publish models to NGC
      description: Publish models to NGC private registry
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
      requestBody:
        content:
          application/json:
            schema: PublishModel
      responses:
        201:
          description: String message for successful upload
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = PublishModel()
    request_schema_data = schema.dump(schema.load(request_data))
    display_name = request_schema_data.get('display_name', '')
    description = request_schema_data.get('description', '')
    team_name = request_schema_data.get('team_name', '')
    # Get response
    response = app_handler.publish_model(org_name, team_name, experiment_id, job_id, display_name=display_name, description=description)
    # Get schema
    schema_dict = None

    if response.code in (200, 201):
        schema = MessageOnlySchema()
        print("Returning success response", response.data, file=sys.stderr)
        schema_dict = schema.dump({"message": "Published model into requested org"})
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
        schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:get_epoch_numbers', methods=['GET'])
@disk_space_check
def experiment_job_get_epoch_numbers(org_name, experiment_id, job_id):
    """Get the epoch numbers for the checkpoints present for this job.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Get epoch numbers present for this job
      description: Get epoch numbers for the checkpoints present for this job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
          description: List of epoch numbers
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_get_epoch_numbers(org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema_dict = None
    if response.code == 200:
        schema = LstIntSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:remove_published_model', methods=['DELETE'])
@disk_space_check
def experiment_remove_published_model(org_name, experiment_id, job_id):
    """Remove published models from NGC.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Remove publish models from NGC
      description: Remove models from NGC private registry
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
      requestBody:
        content:
          application/json:
            schema: PublishModel
      responses:
        201:
          description: String message for successfull deletion
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.args.to_dict()
    schema = PublishModel()
    request_schema_data = schema.dump(schema.load(request_data))
    team_name = request_schema_data.get('team_name', '')
    # Get response
    response = app_handler.remove_published_model(org_name, team_name, experiment_id, job_id)
    # Get schema
    schema_dict = None

    if response.code in (200, 201):
        schema = MessageOnlySchema()
        print("Returning success response", file=sys.stderr)
        schema_dict = schema.dump({"message": "Removed model"})
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
        schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs', methods=['GET'])
@disk_space_check
def experiment_job_list(org_name, experiment_id):
    """List Jobs for Experiment.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Jobs for Experiment
      description: Returns the list of Jobs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
              schema: ExperimentJobListSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=None if experiment_id in ("*", "all") else experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), request.cookies, org_name)

    # Get response
    response = app_handler.job_list(user_id, org_name, experiment_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        filtered_jobs = filtering.apply(request.args, response.data)
        paginated_jobs = pagination.apply(request.args, filtered_jobs)
        metadata = {"jobs": paginated_jobs}
        # Pagination
        skip = request.args.get("skip", None)
        size = request.args.get("size", None)
        if skip is not None and size is not None:
            skip = int(skip)
            size = int(size)
            metadata["pagination_info"] = {
                "total_records": len(filtered_jobs),
                "total_pages": math.ceil(len(filtered_jobs) / size),
                "page_size": size,
                "page_index": skip // size,
            }
        schema = ExperimentJobListSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))))
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def experiment_job_retrieve(org_name, experiment_id, job_id):
    """Retrieve Job for Experiment.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Job for Experiment
      description: Returns the Job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_retrieve(org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentJobSchema()
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/logs', methods=['GET'])
def experiment_job_logs(org_name, experiment_id, job_id):
    """Get realtime job logs. AutoML train job will return current recommendation's experiment log.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Get Job logs for Experiment
      description: Returns the job logs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
      - name: automl_experiment_index
        in: query
        description: Optional filter to retrieve logs from specific autoML experiment
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      responses:
        200:
          description: Returned Job Logs
          content:
            text/plain:
              example: "Execution status: PASS"
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Job not exist or logs not found.
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_job_logs(org_name, experiment_id, job_id, "experiment", request.args.get('automl_experiment_index', None))
    if response.code == 200:
        response = make_response(response.data, 200)
        response.mimetype = 'text/plain'
        return response
    # Handle errors
    schema = ErrorRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(response.data))), 400)
    return response


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:automl_details', methods=['GET'])
@disk_space_check
def experiment_job_automl_details(org_name, experiment_id, job_id):
    """Retrieve AutoML details.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve usable AutoML details
      description: Retrieve usable AutoML details
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    response = app_handler.automl_details(org_name, experiment_id, job_id)
    # Get schema
    schema = AutoMLResultsDetailedSchema()
    if response.code == 200:
        if isinstance(response.data, dict) or response.data == []:
            response = make_response(jsonify(schema.dump(schema.load(response.data))), response.code)
            return response
        metadata = {"error_desc": "internal error: file list invalid", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:status_update', methods=['POST'])
@disk_space_check
def experiment_job_status_update(org_name, experiment_id, job_id):
    """Update Job status for Experiment.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Posts status for the job
      description: Saves recieved content to status file
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = app_handler.job_status_update(org_name, experiment_id, job_id, "experiment", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:log_update', methods=['POST'])
@disk_space_check
def experiment_job_log_update(org_name, experiment_id, job_id):
    """Update Job log for Experiment.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Posts log for the job
      description: Saves recieved content to log file
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = app_handler.job_log_update(org_name, experiment_id, job_id, "experiment", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>:cancel_all_jobs', methods=['POST'])
@disk_space_check
def experiment_jobs_cancel(org_name, experiment_id):
    """Cancel all jobs within experiment (or pause training).
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel all Jobs under experiment
      description: Cancel all Jobs under experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully canceled all jobs under experiments
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.all_job_cancel(org_name, experiment_id, "experiment")
    # Get schema
    if response.code in (200, 201):
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:pause', methods=['POST'])
@disk_space_check
def experiment_job_pause(org_name, experiment_id, job_id):
    """Pause Experiment Job (only for training).
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Pause Experiment Job - only for training
      description: Pause Experiment Job - only for training
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
          description: Successfully requested training pause of specified Job ID (asynchronous)
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_pause(org_name, experiment_id, job_id, "experiment")
    # Get schema
    if response.code in (200, 201):
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def experiment_job_cancel(org_name, experiment_id, job_id):
    """Cancel Experiment Job (or pause training).
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel Experiment Job or pause training
      description: Cancel Experiment Job or pause training
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
          description: Successfully requested cancelation or training pause of specified Job ID
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_cancel(org_name, experiment_id, job_id, "experiment")
    # Get schema
    if response.code in (200, 201):
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def experiment_job_delete(org_name, experiment_id, job_id):
    """Delete Experiment Job.
    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete Experiment Job
      description: Delete Experiment Job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_delete(org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:resume', methods=['POST'])
@disk_space_check
def experiment_job_resume(org_name, experiment_id, job_id):
    """Resume Experiment Job - train/retrain only.
    ---
    post:
      tags:
      - EXPERIMENT
      summary: Resume Experiment Job
      description: Resume Experiment Job - train/retrain only
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
      requestBody:
        content:
          application/json:
            schema: JobResumeSchema
        description: Adjustable metadata for the resumed job.
        required: false
      responses:
        200:
          description: Successfully requested resume of specified Job ID (asynchronous)
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = JobResumeSchema()
    request_schema_data = schema.dump(schema.load(request_data))
    parent_job_id = request_schema_data.get('parent_job_id', None)
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    num_gpu = request_schema_data.get('num_gpu', -1)
    platform = request_schema_data.get('platform', None)
    if parent_job_id:
        parent_job_id = str(parent_job_id)
    specs = request_schema_data.get('specs', {})
    # Get response
    response = app_handler.resume_experiment_job(org_name, experiment_id, job_id, "experiment", parent_job_id, specs=specs, name=name, description=description, num_gpu=num_gpu, platform=platform)
    # Get schema
    if response.code == 200:
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def experiment_job_download(org_name, experiment_id, job_id):
    """Download Job Artifacts.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Download Job Artifacts
      description: Download the Artifacts produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
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
    response = app_handler.job_download(org_name, experiment_id, job_id, "experiment", export_type=export_type.name)
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


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def experiment_job_files_list(org_name, experiment_id, job_id):
    """List Job Files.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Job Files
      description: List the Files produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    retrieve_logs = ast.literal_eval(request.args.get("retrieve_logs", "False"))
    retrieve_specs = ast.literal_eval(request.args.get("retrieve_specs", "False"))
    response = app_handler.job_list_files(org_name, experiment_id, job_id, retrieve_logs, retrieve_specs, "experiment")
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


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:download_selective_files', methods=['GET'])
@disk_space_check
def experiment_job_download_selective_files(org_name, experiment_id, job_id):
    """Download selective Job Artifacts.
    ---
    get:
      tags:
      - EXPERIMENT
      summary: Download selective Job Artifacts
      description: Download selective Artifacts produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
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
    response = app_handler.job_download(org_name, experiment_id, job_id, "experiment", file_lists=file_lists, best_model=best_model, latest_model=latest_model, tar_files=tar_files)
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
    return make_response(jsonify(['api', 'openapi.yaml', 'openapi.json', 'rapipdf', 'redoc', 'swagger', 'version', 'tao_api_notebooks.zip']))


@app.route('/api', methods=['GET'])
def version_list():
    """version list endpoint"""
    return make_response(jsonify(['v1']))


@app.route('/api/v1', methods=['GET'])
def version_v1():
    """version endpoint"""
    return make_response(jsonify(['login', 'user', 'auth', 'health']))


@app.route('/api/v1/orgs', methods=['GET'])
def user_list():
    """user list endpoint"""
    error = {"error_desc": "Listing orgs is not authorized: Missing Org Name", "error_code": 1}
    schema = ErrorRspSchema()
    return make_response(jsonify(schema.dump(schema.load(error))), 403)


@app.route('/api/v1/orgs/<org_name>', methods=['GET'])
@disk_space_check
def user(org_name):
    """user endpoint"""
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


@app.route('/rapipdf', methods=['GET'])
def rapipdf():
    """rapipdf endpoint"""
    return render_template('rapipdf.html')


@app.route('/redoc', methods=['GET'])
def redoc():
    """redoc endpoint"""
    return render_template('redoc.html')


@app.route('/swagger', methods=['GET'])
def swagger():
    """swagger endpoint"""
    return render_template('swagger.html')


@app.route('/version', methods=['GET'])
def version():
    """version endpoint"""
    git_branch = os.environ.get('GIT_BRANCH', 'unknown')
    git_commit_sha = os.environ.get('GIT_COMMIT_SHA', 'unknown')
    git_commit_time = os.environ.get('GIT_COMMIT_TIME', 'unknown')
    version = {'version': tao_version, 'branch': git_branch, 'sha': git_commit_sha, 'time': git_commit_time}
    r = make_response(jsonify(version))
    r.mimetype = 'application/json'
    return r


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
    from handlers.monai_dataset_handler import MonaiDatasetHandler
    MonaiDatasetHandler.clean_cache()


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
    spec.path(view=workspace_list)
    spec.path(view=workspace_retrieve)
    spec.path(view=workspace_retrieve_datasets)
    spec.path(view=workspace_delete)
    spec.path(view=workspace_create)
    spec.path(view=workspace_update)
    spec.path(view=workspace_partial_update)
    spec.path(view=get_dataset_formats)
    spec.path(view=dataset_list)
    spec.path(view=dataset_retrieve)
    spec.path(view=dataset_delete)
    spec.path(view=dataset_create)
    spec.path(view=dataset_update)
    spec.path(view=dataset_partial_update)
    spec.path(view=dataset_specs_schema)
    spec.path(view=dataset_job_run)
    spec.path(view=dataset_job_list)
    spec.path(view=dataset_job_retrieve)
    spec.path(view=dataset_job_logs)
    spec.path(view=dataset_job_cancel)
    spec.path(view=dataset_job_delete)
    spec.path(view=dataset_job_download)
    spec.path(view=experiment_list)
    spec.path(view=base_experiment_list)
    spec.path(view=experiment_retrieve)
    spec.path(view=experiment_delete)
    spec.path(view=experiment_create)
    spec.path(view=experiment_update)
    spec.path(view=experiment_partial_update)
    spec.path(view=experiment_specs_schema)
    spec.path(view=base_experiment_specs_schema)
    spec.path(view=experiment_job_run)
    spec.path(view=experiment_job_get_epoch_numbers)
    spec.path(view=experiment_model_publish)
    spec.path(view=experiment_remove_published_model)
    spec.path(view=experiment_job_list)
    spec.path(view=experiment_job_retrieve)
    spec.path(view=experiment_job_logs)
    spec.path(view=experiment_job_automl_details)
    spec.path(view=experiment_job_pause)
    spec.path(view=experiment_jobs_cancel)
    spec.path(view=experiment_job_cancel)
    spec.path(view=experiment_job_delete)
    spec.path(view=experiment_job_resume)
    spec.path(view=experiment_job_download)


if __name__ == '__main__':
    time_loop.start()

    # app.run(host='0.0.0.0', port=8000)
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        app.run(host="0.0.0.0", port=8008)
    else:
        app.run()
