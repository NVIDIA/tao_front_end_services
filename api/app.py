#!/usr/bin/env python3

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

"""API modules defining schemas and endpoints"""
import ast
import sys
import math
import json
import shutil

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from flask import Flask, request, jsonify, make_response, render_template, send_from_directory, send_file
from marshmallow import Schema, fields
from marshmallow_enum import EnumField, Enum

from filter_utils import filtering, pagination
from auth_utils import credentials, authentication, access_control
from health_utils import health_check

from handlers.app_handler import AppHandler as app_handler
from handlers.utilities import validate_uuid
from job_utils.workflow import Workflow

from werkzeug.exceptions import HTTPException


flask_plugin = FlaskPlugin()
marshmallow_plugin = MarshmallowPlugin()


#
# Create an APISpec
#
spec = APISpec(
    title='TAO Toolkit API',
    version='v5.2.0',
    openapi_version='3.0.3',
    info={"description": 'TAO Toolkit API document'},
    tags=[
        {"name": 'AUTHENTICATION', "description": 'Endpoints related to User Authentication'},
        {"name": 'DATASET', "description": 'Endpoints related to Datasets'},
        {"name": 'MODEL', "description": 'Endpoints related to Model Experiments'}
    ],
    plugins=[flask_plugin, marshmallow_plugin],
)


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

    message = fields.Str(allow_none=True)


class ErrorRspSchema(Schema):
    """Class defining error response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    error_desc = fields.Str()
    error_code = fields.Int()


class JobStatusEnum(Enum):
    """Class defining job status enum"""

    Done = 'Done'
    Running = 'Running'
    Error = 'Error'
    Pending = "Pending"


#
# Flask app
#
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['TRAP_HTTP_EXCEPTIONS'] = True


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
    date = fields.Str()
    time = fields.Str()
    message = fields.Str()
    status = fields.Str()


class GraphSchema(Schema):
    """Class defining Graph schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True)
    x_min = fields.Int(allow_none=True)
    x_max = fields.Int(allow_none=True)
    y_min = fields.Float(allow_none=True)
    y_max = fields.Float(allow_none=True)
    values = fields.Dict(keys=fields.Int(allow_none=True), values=fields.Float(allow_none=True))
    units = fields.Str(allow_none=True)


class CategoryWiseSchema(Schema):
    """Class defining CategoryWise schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    category = fields.Str()
    value = fields.Float(allow_none=True)


class CategorySchema(Schema):
    """Class defining Category schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str()
    category_wise_values = fields.List(fields.Nested(CategoryWiseSchema, allow_none=True))


class KPISchema(Schema):
    """Class defining KPI schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True)
    values = fields.Dict(keys=fields.Int(allow_none=True), values=fields.Float(allow_none=True))


class CustomFloatField(fields.Float):
    """Class defining custom Float field allown NaN and Inf values in Marshmallow"""

    def _deserialize(self, value, attr, data, **kwargs):
        if value == "nan" or (type(value) == float and math.isnan(value)):
            return float("nan")
        if value == "inf" or (type(value) == float and math.isinf(value)):
            return float("inf")
        if value == "-inf" or (type(value) == float and math.isinf(value)):
            return float("-inf")
        if value is None:
            return None
        return super()._deserialize(value, attr, data)


class AutoMLResultsSchema(Schema):
    """Class defining AutoML results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True)
    value = CustomFloatField()


class StatsSchema(Schema):
    """Class defining results stats schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True)
    value = fields.Str(allow_none=True)


class JobResultSchema(Schema):
    """Class defining job results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    detailed_status = fields.Nested(DetailedStatusSchema, allow_none=True)
    graphical = fields.List(fields.Nested(GraphSchema, allow_none=True))
    categorical = fields.List(fields.Nested(CategorySchema, allow_none=True))
    kpi = fields.List(fields.Nested(KPISchema, allow_none=True))
    automl_result = fields.List(fields.Nested(AutoMLResultsSchema, allow_none=True))
    stats = fields.List(fields.Nested(StatsSchema, allow_none=True))
    epoch = fields.Int(allow_none=True)
    max_epoch = fields.Int(allow_none=True)
    time_per_epoch = fields.Str(allow_none=True)
    time_per_iter = fields.Str(allow_none=True)
    cur_iter = fields.Int(allow_none=True)
    eta = fields.Str(allow_none=True)


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
    ngc_api_key = fields.Str()


class LoginRspSchema(Schema):
    """Class defining login response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    user_id = fields.UUID()
    token = fields.Str()


@app.route('/api/v1/login', methods=['POST'])
def login():
    """User Login.
    ---
    post:
      tags:
      - AUTHENTICATION
      summary: User Login
      description: Returns the user credentials
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
        401:
          description: Unauthorized
          content:
            application/json:
              schema: ErrorRspSchema
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
def auth():
    """authentication endpoint"""
    # retrieve jwt from headers
    token = ''
    url = request.headers.get('X-Original-Url', '')
    print('URL: ' + str(url), flush=True)
    authorization = request.headers.get('Authorization', '')
    authorization_parts = authorization.split()
    if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'bearer':
        token = authorization_parts[1]
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


#
# DATASET API
#
class DatasetActions(Schema):
    """Class defining dataset actions schema"""

    job = fields.UUID(allow_none=True)
    actions = fields.List(fields.Str())
    specs = fields.Raw()
    parent_id = fields.UUID(allow_none=True)
    parent_job_type = fields.Str(allow_none=True)


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


class DatasetReqSchema(Schema):
    """Class defining dataset request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    name = fields.Str()
    description = fields.Str()
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(allow_none=True))
    version = fields.Str()
    logo = fields.URL()
    type = EnumField(DatasetTypeEnum)
    format = EnumField(DatasetFormatEnum)
    pull = fields.URL()


class DatasetJobResultCategoriesSchema(Schema):
    """Class defining dataset job result categories schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    category = fields.Str()
    count = fields.Int()


class DatasetJobResultTotalSchema(Schema):
    """Class defining dataset job result total schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    images = fields.Int()
    labels = fields.Int()


class DatasetJobSchema(Schema):
    """Class defining dataset job result total schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    id = fields.UUID()
    parent_id = fields.UUID(allow_none=True)
    created_on = fields.DateTime()
    last_modified = fields.DateTime()
    action = fields.Str()
    status = EnumField(JobStatusEnum)
    result = fields.Nested(JobResultSchema)


class DatasetRspSchema(Schema):
    """Class defining dataset response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    id = fields.UUID()
    created_on = fields.DateTime()
    last_modified = fields.DateTime()
    name = fields.Str()
    description = fields.Str()
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(allow_none=True))
    version = fields.Str()
    logo = fields.URL(allow_none=True)
    type = EnumField(DatasetTypeEnum)
    format = EnumField(DatasetFormatEnum)
    pull = fields.URL(allow_none=True)
    actions = fields.List(fields.Str())
    jobs = fields.List(fields.Nested(DatasetJobSchema))


class DatasetListRspSchema(Schema):
    """Class defining dataset list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    datasets = fields.List(fields.Nested(DatasetRspSchema))


class DatasetJobListSchema(Schema):
    """Class defining dataset list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    jobs = fields.List(fields.Nested(DatasetJobSchema))


@app.route('/api/v1/user/<user_id>/dataset', methods=['GET'])
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
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>', methods=['GET'])
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
      - name: dataset_id
        in: path
        description: ID of Dataset to return
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Dataset
          content:
            application/json:
              schema: DatasetRspSchema
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>', methods=['DELETE'])
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
      - name: dataset_id
        in: path
        description: ID of Dataset to delete
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Deleted Dataset
          content:
            application/json:
              schema: DatasetRspSchema
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset', methods=['POST'])
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
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>', methods=['PUT'])
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
      - name: dataset_id
        in: path
        description: ID of Dataset to update
        required: true
        schema:
          type: string
          format: uuid
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
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>', methods=['PATCH'])
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
      - name: dataset_id
        in: path
        description: ID of Dataset to update
        required: true
        schema:
          type: string
          format: uuid
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
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route("/api/v1/user/<user_id>/dataset/<dataset_id>/upload", methods=["POST"])
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
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
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
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    file_tgz = request.files.get("file", None)
    # Get response
    print("Triggering API call to upload data to server", file=sys.stderr)
    response = app_handler.upload_dataset(user_id, dataset_id, file_tgz)
    print("API call to upload data to server complete", file=sys.stderr)
    # Get schema
    schema_dict = None
    if response.code == 201:
        schema = DatasetUploadSchema()
        print("Returning success response", file=sys.stderr)
        schema_dict = schema.dump({"message": "Data successfully uploaded"})
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
        schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/specs/<action>/schema', methods=['GET'])
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
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
        404:
          description: User, Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/specs/<action>', methods=['GET'])
def dataset_specs_retrieve(user_id, dataset_id, action):
    """Retrieve Dataset Specs.
    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Dataset Specs
      description: Returns the saved Dataset Specs for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      responses:
        200:
          description: Returned the saved Dataset Specs for specified action
          content:
            application/json:
              schema:
                type: object
        404:
          description: User, Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_spec(user_id, dataset_id, action, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/specs/<action>', methods=['POST'])
def dataset_specs_save(user_id, dataset_id, action):
    """Save Dataset Specs.
    ---
    post:
      tags:
      - DATASET
      summary: Save Dataset Specs
      description: Save the Dataset Specs for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      requestBody:
        content:
          application/json:
            schema:
              type: object
        description: Dataset Specs
        required: true
      responses:
        201:
          description: Returned the saved Dataset Specs for specified action
          content:
            application/json:
              schema:
                type: object
        400:
          description: Invalid specs
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User, Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_dict = request.get_json(force=True)
    # Get response
    response = app_handler.save_spec(user_id, dataset_id, action, request_dict, "dataset")
    # Get schema
    schema = None
    if response.code == 201:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/specs/<action>', methods=['PUT'])
def dataset_specs_update(user_id, dataset_id, action):
    """Update Dataset Specs.
    ---
    put:
      tags:
      - DATASET
      summary: Update Dataset Specs
      description: Update the Dataset Specs for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      requestBody:
        content:
          application/json:
            schema:
              type: object
        description: Dataset Specs
        required: true
      responses:
        200:
          description: Returned the updated Dataset Specs for specified action
          content:
            application/json:
              schema:
                type: object
        400:
          description: Invalid specs
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User, Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_dict = request.get_json(force=True)
    # Get response
    response = app_handler.update_spec(user_id, dataset_id, action, request_dict, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/job', methods=['POST'])
def dataset_job_run(user_id, dataset_id):
    """Run Dataset Jobs.
    ---
    post:
      tags:
      - DATASET
      summary: Run Dataset Jobs
      description: Asynchronously starts a list of Dataset Actions and returns corresponding Job IDs
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
      requestBody:
        content:
          application/json:
            schema:
              type: array
              items: DatasetActions
      responses:
        201:
          description: Returned the list of Job IDs corresponding to requested Dataset Actions
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
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
    requested_job = request_schema_data.get('job', None)
    if requested_job:
        requested_job = str(requested_job)
    parent_handler_id = request_schema_data.get('parent_id', None)
    if parent_handler_id:
        parent_handler_id = str(parent_handler_id)
    parent_job_type = request_schema_data.get('parent_job_type', None)
    requested_actions = request_schema_data.get('actions', [])
    specs = request_schema_data.get('specs', {})
    # Get response
    response = app_handler.job_run(user_id, dataset_id, parent_handler_id, requested_job, requested_actions, "dataset", parent_job_type, specs=specs)
    # Get schema
    if response.code == 201:
        if isinstance(response.data, list) and all(not validate_uuid(job_id=j) for j in response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/job', methods=['GET'])
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
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
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
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, dataset_id=dataset_id)
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/job/<job_id>', methods=['GET'])
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
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Job
          content:
            application/json:
              schema: DatasetJobSchema
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/job/<job_id>/cancel', methods=['POST'])
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
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Successfully requested cancelation of specified Job ID (asynchronous)
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/job/<job_id>', methods=['DELETE'])
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
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Successfully requested deletion of specified Job ID
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/job/<job_id>/list_files', methods=['GET'])
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
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Job Files
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/job/<job_id>/download_selective_files', methods=['GET'])
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
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
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


@app.route('/api/v1/user/<user_id>/dataset/<dataset_id>/job/<job_id>/download', methods=['GET'])
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
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
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
# MODEL API
#
class ModelActions(Schema):
    """Class defining model actions schema"""

    job = fields.UUID(allow_none=True)
    actions = fields.List(fields.Str())
    specs = fields.Raw()
    parent_id = fields.UUID(allow_none=True)
    parent_job_type = fields.Str(allow_none=True)


class CheckpointChooseMethodEnum(Enum):
    """Class defining enum for methods of picking a trained checkpoint"""

    latest_model = 'latest_model'
    best_model = 'best_model'
    from_epoch_number = 'from_epoch_number'


class ModelNetworkArchEnum(Enum):
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


class ModelReqSchema(Schema):
    """Class defining model request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    name = fields.Str()
    description = fields.Str()
    version = fields.Str()
    logo = fields.URL()
    ngc_path = fields.Str()
    additional_id_info = fields.Str()
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(allow_none=True))
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(keys=fields.Str(allow_none=True), values=fields.Int(allow_none=True))
    encryption_key = fields.Str()
    network_arch = EnumField(ModelNetworkArchEnum)
    ptm = fields.List(fields.UUID())
    eval_dataset = fields.UUID()
    inference_dataset = fields.UUID()
    calibration_dataset = fields.UUID()
    train_datasets = fields.List(fields.UUID())
    read_only = fields.Bool()
    public = fields.Bool()
    automl_enabled = fields.Bool(allow_none=True)
    automl_algorithm = fields.Str(allow_none=True)
    automl_max_recommendations = fields.Int(allow_none=True)
    automl_delete_intermediate_ckpt = fields.Bool(allow_none=True)
    automl_R = fields.Int(allow_none=True)
    automl_nu = fields.Int(allow_none=True)
    metric = fields.Str(allow_none=True)
    epoch_multiplier = fields.Int(allow_none=True)
    automl_add_hyperparameters = fields.Str(allow_none=True)
    automl_remove_hyperparameters = fields.Str(allow_none=True)


class ModelJobSchema(Schema):
    """Class defining model job schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    id = fields.UUID()
    parent_id = fields.UUID(allow_none=True)
    created_on = fields.DateTime()
    last_modified = fields.DateTime()
    action = fields.Str()
    status = EnumField(JobStatusEnum)
    result = fields.Nested(JobResultSchema)


class ModelRspSchema(Schema):
    """Class defining model respnse schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    id = fields.UUID()
    created_on = fields.DateTime()
    last_modified = fields.DateTime()
    name = fields.Str()
    description = fields.Str()
    version = fields.Str()
    logo = fields.URL(allow_none=True)
    ngc_path = fields.Str(allow_none=True)
    additional_id_info = fields.Str(allow_none=True)
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(allow_none=True))
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(keys=fields.Str(allow_none=True), values=fields.Int(allow_none=True))
    encryption_key = fields.Str()
    network_arch = EnumField(ModelNetworkArchEnum)
    ptm = fields.List(fields.UUID())
    dataset_type = EnumField(DatasetTypeEnum)
    eval_dataset = fields.UUID(allow_none=True)
    inference_dataset = fields.UUID(allow_none=True)
    calibration_dataset = fields.UUID(allow_none=True)
    train_datasets = fields.List(fields.UUID())
    read_only = fields.Bool()
    public = fields.Bool()
    actions = fields.List(fields.Str())
    jobs = fields.List(fields.Nested(ModelJobSchema))
    automl_enabled = fields.Bool(allow_none=True)
    automl_algorithm = fields.Str(allow_none=True)
    automl_max_recommendations = fields.Int(allow_none=True)
    automl_delete_intermediate_ckpt = fields.Bool(allow_none=True)
    automl_R = fields.Int(allow_none=True)
    automl_nu = fields.Int(allow_none=True)
    metric = fields.Str(allow_none=True)
    epoch_multiplier = fields.Int(allow_none=True)
    automl_add_hyperparameters = fields.Str(allow_none=True)
    automl_remove_hyperparameters = fields.Str(allow_none=True)


class ModelListRspSchema(Schema):
    """Class defining model list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    models = fields.List(fields.Nested(ModelRspSchema))


class ModelJobListSchema(Schema):
    """Class defining model job list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    jobs = fields.List(fields.Nested(ModelJobSchema))


@app.route('/api/v1/user/<user_id>/model', methods=['GET'])
def model_list(user_id):
    """List Models.
    ---
    get:
      tags:
      - MODEL
      summary: List Models
      description: Returns the list of Models
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
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
      responses:
        200:
          description: Returned the list of Models
          content:
            application/json:
              schema:
                type: array
                items: ModelRspSchema
    """
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    models = app_handler.list_models(user_id)
    filtered_models = filtering.apply(request.args, models)
    paginated_models = pagination.apply(request.args, filtered_models)
    pagination_total = len(filtered_models)
    metadata = {"models": paginated_models}
    schema = ModelListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))['models']))
    response.headers['X-Pagination-Total'] = str(pagination_total)
    return response


@app.route('/api/v1/user/<user_id>/model/<model_id>', methods=['GET'])
def model_retrieve(user_id, model_id):
    """Retrieve Model.
    ---
    get:
      tags:
      - MODEL
      summary: Retrieve Model
      description: Returns the Model
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID of Model to return
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned the Model
          content:
            application/json:
              schema: ModelRspSchema
        404:
          description: User or Model not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.retrieve_model(user_id, model_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ModelRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>', methods=['DELETE'])
def model_delete(user_id, model_id):
    """Delete Model.
    ---
    delete:
      tags:
      - MODEL
      summary: Delete Model
      description: Cancels all related running jobs and returns the deleted Model
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID of Model to delete
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned the deleted Model
          content:
            application/json:
              schema: ModelRspSchema
        404:
          description: User or Model not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_model(user_id, model_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ModelRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model', methods=['POST'])
def model_create(user_id):
    """Create new Model.
    ---
    post:
      tags:
      - MODEL
      summary: Create new Model
      description: Returns the new Model
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      requestBody:
        content:
          application/json:
            schema: ModelReqSchema
        description: Initial metadata for new Model (ptm or network_arch required)
        required: true
      responses:
        201:
          description: Returned the new Model
          content:
            application/json:
              schema: ModelRspSchema
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ModelReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.create_model(user_id, request_dict)
    # Get schema
    schema = None
    if response.code == 201:
        schema = ModelRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>', methods=['PUT'])
def model_update(user_id, model_id):
    """Update Model.
    ---
    put:
      tags:
      - MODEL
      summary: Update Model
      description: Returns the updated Model
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID of Model to update
        required: true
        schema:
          type: string
          format: uuid
      requestBody:
        content:
          application/json:
            schema: ModelReqSchema
        description: Updated metadata for Model
        required: true
      responses:
        200:
          description: Returned the updated Model
          content:
            application/json:
              schema: ModelRspSchema
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User or Model not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ModelReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_model(user_id, model_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ModelRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>', methods=['PATCH'])
def model_partial_update(user_id, model_id):
    """Partial update Model.
    ---
    patch:
      tags:
      - MODEL
      summary: Partial update Model
      description: Returns the updated Model
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID of Model to update
        required: true
        schema:
          type: string
          format: uuid
      requestBody:
        content:
          application/json:
            schema: ModelReqSchema
        description: Updated metadata for Model
        required: true
      responses:
        200:
          description: Returned the updated Model
          content:
            application/json:
              schema: ModelRspSchema
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User or Model not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ModelReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_model(user_id, model_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ModelRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/specs/<action>/schema', methods=['GET'])
def specs_schema_without_handler_id(user_id, action):
    """Retrieve Specs schema.
    ---
    get:
      summary: Retrieve Specs schema without model or dataset id
      description: Returns the Specs schema for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      responses:
        200:
          description: Returned the Specs schema for given action and network
          content:
            application/json:
              schema:
                type: object
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
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

    response = app_handler.get_spec_schema_without_handler_id(network, format, action, train_datasets)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/specs/<action>/schema', methods=['GET'])
def model_specs_schema(user_id, model_id, action):
    """Retrieve Specs schema.
    ---
    get:
      tags:
      - MODEL
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
      - name: model_id
        in: path
        description: ID for Model
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_spec_schema(user_id, model_id, action, "model")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/specs/<action>', methods=['GET'])
def model_specs_retrieve(user_id, model_id, action):
    """Retrieve Model Specs.
    ---
    get:
      tags:
      - MODEL
      summary: Retrieve Model Specs
      description: Returns the Model Specs for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID of Model
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      responses:
        200:
          description: Returned the Model Specs for specified action
          content:
            application/json:
              schema:
                type: object
        404:
          description: User, Model or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_spec(user_id, model_id, action, "model")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/specs/<action>', methods=['POST'])
def model_specs_save(user_id, model_id, action):
    """Save Model Specs.
    ---
    post:
      tags:
      - MODEL
      summary: Save Model Specs
      description: Save the Model Specs for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID of Model
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      requestBody:
        content:
          application/json:
            schema:
              type: object
        description: Model Specs
        required: true
      responses:
        201:
          description: Returned the saved Model Specs for specified action
          content:
            application/json:
              schema:
                type: object
        400:
          description: Invalid specs
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User, Model or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_dict = request.get_json(force=True)
    # Get response
    response = app_handler.save_spec(user_id, model_id, action, request_dict, "model")
    # Get schema
    schema = None
    if response.code == 201:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/specs/<action>', methods=['PUT'])
def model_specs_update(user_id, model_id, action):
    """Update Model Specs.
    ---
    put:
      tags:
      - MODEL
      summary: Update Model Specs
      description: Update the Model Specs for a given action
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID of Model
        required: true
        schema:
          type: string
          format: uuid
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
      requestBody:
        content:
          application/json:
            schema:
              type: object
        description: Model Specs
        required: true
      responses:
        200:
          description: Returned the updated Model Specs for specified action
          content:
            application/json:
              schema:
                type: object
        400:
          description: Invalid specs
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: User, Model or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_dict = request.get_json(force=True)
    # Get response
    response = app_handler.save_spec(user_id, model_id, action, request_dict, "model")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/job', methods=['POST'])
def model_job_run(user_id, model_id):
    """Run Model Jobs.
    ---
    post:
      tags:
      - MODEL
      summary: Run Model Jobs
      description: Asynchronously starts a list of Model Actions and returns corresponding Job IDs
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID for Model
        required: true
        schema:
          type: string
          format: uuid
      requestBody:
        content:
          application/json:
            schema:
              type: array
              items: ModelActions
      responses:
        201:
          description: Returned the list of Job IDs corresponding to requested Model Actions
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
        404:
          description: User or Model not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = ModelActions()
    request_schema_data = schema.dump(schema.load(request_data))
    requested_job = request_schema_data.get('job', None)
    if requested_job:
        requested_job = str(requested_job)
    parent_handler_id = request_schema_data.get('parent_id', None)
    if parent_handler_id:
        parent_handler_id = str(parent_handler_id)
    parent_job_type = request_schema_data.get('parent_job_type', None)
    requested_actions = request_schema_data.get('actions', [])
    specs = request_schema_data.get('specs', {})
    # Get response
    response = app_handler.job_run(user_id, model_id, parent_handler_id, requested_job, requested_actions, "model", parent_job_type, specs=specs)
    # Get schema
    schema = None
    if response.code == 201:
        if isinstance(response.data, list) and all(not validate_uuid(job_id=j) for j in response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/job', methods=['GET'])
def model_job_list(user_id, model_id):
    """List Jobs for Model.
    ---
    get:
      tags:
      - MODEL
      summary: List Jobs for Model
      description: Returns the list of Jobs
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID for Model
        required: true
        schema:
          type: string
          format: uuid
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
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
                items: ModelJobSchema
        404:
          description: User or Model not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_list(user_id, model_id, "model")
    # Get schema
    schema = None
    if response.code == 200:
        pagination_total = 0
        metadata = {"jobs": response.data}
        schema = ModelJobListSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))['jobs']))
        response.headers['X-Pagination-Total'] = str(pagination_total)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/job/<job_id>', methods=['GET'])
def model_job_retrieve(user_id, model_id, job_id):
    """Retrieve Job for Model.
    ---
    get:
      tags:
      - MODEL
      summary: Retrieve Job for Model
      description: Returns the Job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID of Model
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Job
          content:
            application/json:
              schema: ModelJobSchema
        404:
          description: User, Model or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_retrieve(user_id, model_id, job_id, "model")
    # Get schema
    schema = None
    if response.code == 200:
        schema = ModelJobSchema()
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/job/<job_id>/cancel', methods=['POST'])
def model_job_cancel(user_id, model_id, job_id):
    """Cancel Model Job (or pause training).
    ---
    post:
      tags:
      - MODEL
      summary: Cancel Model Job or pause training
      description: Cancel Model Job or pause training
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID for Model
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Successfully requested cancelation or training pause of specified Job ID (asynchronous)
        404:
          description: User, Model or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_cancel(user_id, model_id, job_id, "model")
    # Get schema
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/job/<job_id>', methods=['DELETE'])
def model_job_delete(user_id, model_id, job_id):
    """Delete Model Job.
    ---
    post:
      tags:
      - MODEL
      summary: Delete Model Job
      description: Delete Model Job
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID for Model
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Successfully requested deletion of specified Job ID
        404:
          description: User, Model or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_delete(user_id, model_id, job_id, "model")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/job/<job_id>/resume', methods=['POST'])
def model_job_resume(user_id, model_id, job_id):
    """Resume Model Job - train/retrain only.
    ---
    post:
      tags:
      - MODEL
      summary: Resume Model Job
      description: Resume Model Job - train/retrain only
      parameters:
      - name: user_id
        in: path
        description: User ID
        required: true
        schema:
          type: string
          format: uuid
      - name: model_id
        in: path
        description: ID for Model
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Successfully requested resume of specified Job ID (asynchronous)
        404:
          description: User, Model or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = ModelActions()
    request_schema_data = schema.dump(schema.load(request_data))
    requested_job = request_schema_data.get('job', None)
    if requested_job:
        requested_job = str(requested_job)
    specs = request_schema_data.get('specs', {})
    # Get response
    response = app_handler.resume_model_job(user_id, model_id, job_id, "model", specs=specs)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/user/<user_id>/model/<model_id>/job/<job_id>/download', methods=['GET'])
def model_job_download(user_id, model_id, job_id):
    """Download Job Artifacts.
    ---
    get:
      tags:
      - MODEL
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
      - name: model_id
        in: path
        description: ID of Model
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
        404:
          description: User, Model or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_download(user_id, model_id, job_id, "model")
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


@app.route('/api/v1/user/<user_id>/model/<model_id>/job/<job_id>/list_files', methods=['GET'])
def model_job_files_list(user_id, model_id, job_id):
    """List Job Files.
    ---
    get:
      tags:
      - MODEL
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
      - name: model_id
        in: path
        description: ID of Model
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Job Files
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
        404:
          description: User, Model or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    retrieve_logs = ast.literal_eval(request.args.get("retrieve_logs", "False"))
    retrieve_specs = ast.literal_eval(request.args.get("retrieve_specs", "False"))
    response = app_handler.job_list_files(user_id, model_id, job_id, retrieve_logs, retrieve_specs, "model")
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


@app.route('/api/v1/user/<user_id>/model/<model_id>/job/<job_id>/download_selective_files', methods=['GET'])
def model_job_download_selective_files(user_id, model_id, job_id):
    """Download selective Job Artifacts.
    ---
    get:
      tags:
      - MODEL
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
      - name: model_id
        in: path
        description: ID of Model
        required: true
        schema:
          type: string
          format: uuid
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
        404:
          description: User, Model or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(user_id=user_id, model_id=model_id, job_id=job_id)
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
    response = app_handler.job_download(user_id, model_id, job_id, "model", file_lists=file_lists, best_model=best_model, latest_model=latest_model, tar_files=tar_files)
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
def liveness():
    """api liveness endpoint"""
    live_state = health_check.check_logging()
    if live_state:
        return make_response(jsonify("OK"), 201)
    return make_response(jsonify("Error"), 400)


@app.route('/api/v1/health/readiness', methods=['GET'])
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


@app.route('/api/v1/user', methods=['GET'])
def user_list():
    """user list endpoint"""
    error = {"error_desc": "Listing users is not authorized: Missing User ID", "error_code": 1}
    schema = ErrorRspSchema()
    return make_response(jsonify(schema.dump(schema.load(error))), 403)


@app.route('/api/v1/user/<user_id>', methods=['GET'])
def user(user_id):
    """user endpoint"""
    message = validate_uuid(user_id=user_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    return make_response(jsonify(['dataset', 'model']))


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
def download_folder():
    """Download notebooks endpoint"""
    # Create a temporary zip file containing the folder
    shutil.make_archive("/tmp/tao_api_notebooks", 'zip', "notebooks/")

    # Send the zip file for download
    return send_file(
        "/tmp/tao_api_notebooks.zip",
        as_attachment=True,
        download_name="tao_api_notebooks.zip"
    )


#
# End of APIs
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
    spec.path(view=dataset_specs_retrieve)
    spec.path(view=dataset_specs_save)
    spec.path(view=dataset_specs_update)
    spec.path(view=dataset_job_run)
    spec.path(view=dataset_job_list)
    spec.path(view=dataset_job_retrieve)
    spec.path(view=dataset_job_cancel)
    spec.path(view=dataset_job_delete)
    spec.path(view=dataset_job_download)
    spec.path(view=model_list)
    spec.path(view=model_retrieve)
    spec.path(view=model_delete)
    spec.path(view=model_create)
    spec.path(view=model_update)
    spec.path(view=model_partial_update)
    spec.path(view=model_specs_schema)
    spec.path(view=model_specs_retrieve)
    spec.path(view=model_specs_save)
    spec.path(view=model_specs_update)
    spec.path(view=model_job_run)
    spec.path(view=model_job_list)
    spec.path(view=model_job_retrieve)
    spec.path(view=model_job_cancel)
    spec.path(view=model_job_delete)
    spec.path(view=model_job_resume)
    spec.path(view=model_job_download)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000)
    app.run()
