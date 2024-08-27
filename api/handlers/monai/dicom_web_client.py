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

"""Medical communication toolbox"""

import requests
import json

from handlers.utilities import Code, TAOResponse


class DicomWebClient:
    """Class to communicate with the dicom web."""

    successful_code = [200, 201]
    image_type = ["CT", "MR"]

    def __init__(self, location, user_id, access_key):
        """Initialize the DicomWebClient class
        Args:
            location: the url of the dicom web server.
            user_id: the user_id to get access to the dicom web server.
            access_key: the access_key to get access to the dicom web server.
        """
        self.location = location
        self.user_id = user_id
        self.access_key = access_key

    def _get_content(self, content, dicom_filter=None):
        """
        Private method to get the content from the dicom web server.
        Parameters:
            content: the url of the content that makes up a whole url with the self.location url.
            dicom_filter: the filer to get samples from dicom web server.

        Returns:
            201: Got the content successfully.
            error code: The response code from the dicom web server if failed to connect to the server.
        """
        content_url = f"{self.location}/{content}"
        if dicom_filter is None:
            response = requests.get(content_url, auth=(self.user_id, self.access_key))
        if response.status_code not in self.successful_code:
            msg = f"Cannot get {content_url}"
            return Code(response.status_code, {}, msg)
        return Code(201, response.json(), "Got the content.")

    def _get_content_data(self, content, dicom_filter=None):
        """
        Return the content of json if the data is fetched. Otherwise, return error code.
        Parameters:
            content: the url of the content that makes up a whole url with the self.location url.
        """
        content_code = self._get_content(content, dicom_filter)
        if content_code.code in self.successful_code:
            return content_code.data
        return content_code

    def generate_segmentation_sample(self, study_url, image_type, dicom_filter):
        """
        Generate the segmentation sample given the study url. Now this function only
        supports the CT and MR type images.
        # TODO This function should be extended to support more than 2 series and more
        image modalities.
        Parameters:
            study_url: the url of the specific study.
            image_type: the modality in the study that should be treat as images.
            dicom_filter: the filer to get samples from dicom web server.

        Returns:
            image-label pair: if successfully pulling samples from dicom web server.
            error code: if something wrong happened when pulling samples.
        """
        sample = None
        dicom_web_study_dict = self._get_content_data(study_url, dicom_filter)
        # When met something wrong, here will be a TAOResponse to return the error code.
        if isinstance(dicom_web_study_dict, TAOResponse):
            return dicom_web_study_dict
        dicom_web_series = dicom_web_study_dict.get("Series", dicom_filter)
        # If the study is empty, just return nothing.
        if dicom_web_series is None:
            return None

        sample = {"study": study_url, "image": "", "label": "", "update_time": ""}
        # TODO Need to support multi series and update in backend.
        for series in dicom_web_series:
            series_url = f"series/{series}"
            series_dict = self._get_content_data(series_url, dicom_filter)
            if isinstance(series_dict, TAOResponse):
                return series_dict
            if series_dict["MainDicomTags"]["Modality"] in image_type:
                sample["image"] = series_url
            elif series_dict["MainDicomTags"]["Modality"] == "SEG":
                sample["label"] = series_url
        return sample

    def generate_dataset_manifest_dict(self, annotation_type="SEG", dicom_filter=None):
        """
        Generate a manifest dict according to the current dicom web server. Here is what it contains:
            location: the uri root location of the dataset.
            user_id: the user_id to get access to the dataset server.
            access_key: the access_key to get access to the dataset server.
            data: a list contains informations of all samples in the dataset.
            force_fetch: bool that decides whether to fetch the labeled samples.
            labeled_list: a list contains info about labeled samples.
            unlabeled_list: a list contains info about unlabeled samples.

        An example sample in the data list is like:
            {"study": study_url, "image": image_url, "label": image_url, "update_time": xxx-xxx}

        An example sample info in the labeled/unlabeled/fetched lists is like:
            {"index": 0, "fetch_time": 123.1212, "al_score": 0.0}
            where the index indicates the index in the manifest["data"] list.
        """
        manifest_dict = {"location": self.location,
                         "user_id": self.user_id,
                         "access_key": self.access_key,
                         "data": [],
                         "force_fetch": False,
                         "labeled_list": [],
                         "unlabeled_list": []}
        current_url = "studies"
        dicom_web_studies = self._get_content_data(current_url, dicom_filter)
        # When met something wrong, here will be a TAOResponse to return the error code.
        if isinstance(dicom_web_studies, TAOResponse):
            return dicom_web_studies
        for dicom_web_study in dicom_web_studies:
            current_url = f"studies/{dicom_web_study}"
            if annotation_type == "SEG":
                sample = self.generate_segmentation_sample(current_url, self.image_type, dicom_filter)
                if isinstance(sample, TAOResponse):
                    return sample
                sample_index = len(manifest_dict["data"])
                index_info = {"index": sample_index, "fetch_time": 0.0, "al_score": 0.0}
                if sample["label"]:
                    manifest_dict["labeled_list"].append(index_info)
                else:
                    manifest_dict["unlabeled_list"].append(index_info)
                manifest_dict["data"].append(sample)
            else:
                return Code(400, {}, f"Doesn't support annotation type {annotation_type}.")
        return Code(201, manifest_dict, "Got the dict.")

    def create_dataset_manifest_file(self, manifest_path, annotation_type="SEG", dicom_filter=None):
        """
        Create the dataset manifest file to the given path. The manifest file has samples from the
        dicom web server.
        Parameters:
            manifest_path: the path to save the manifest_file.
            annotation_type: what kind of annotation this manifest should have.
            dicom_filter: the filter to get the specific type of samples.

        Returns:
            400: If something wrong happened when saving the manifest file.
            201: Successfully saved the manifest file.
        """
        manifest_dict_status = self.generate_dataset_manifest_dict(annotation_type, dicom_filter)
        if manifest_dict_status.code == 201:
            manifest_dict = manifest_dict_status.data
        else:
            return manifest_dict_status

        try:
            with open(manifest_path, "w", encoding='utf-8') as f:
                f.write(json.dumps(manifest_dict, indent=4))
        except:
            return Code(400, {}, f"Cannot write the {manifest_path}.")

        return Code(201, {}, "Saved the manifest file.")
