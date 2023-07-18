---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Describe the bug**

A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**

Please list *minimal* steps or code snippet for us to be able to reproduce the bug.

A  helpful guide on on how to craft a minimal bug report  http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports. 


**Expected behavior**

A clear and concise description of what you expected to happen.

**Environment overview (please complete the following information)**

 - Environment location: [Bare-metal, Cloud(specify cloud provider - AWS, Azure, GCP, Collab)]
 - Save the logs of the following commands and share them:
   - `kubectl get pods -A`
   - `kubectl get services`
   - `helm list -A`
   - `kubectl get pods | grep nvidia-driver` and `kubectl exec -it <driver_pod> -n nvidia_gpu_operator -- nvidia-smi`
   - `kubectl describe pods <api_pod>`
   - `kubectl logs <api_pod>`
   - `kubectl describe pods <workflow_pod>`
   - `kubectl logs <workflow_pod>`

**Additional context**

Add any other context about the problem here.
Example: Any spec values overriden, multi-gpu, notebook and model trying to execute