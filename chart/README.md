# Helm chart for deploying NVIDIA Transfer Learning API 

## Prerequisites
- Hardware setup (GPU Node(s) with Ubuntu 18.04 or latest)
- Kubernetes cluster 
- NVIDIA GPU Operator (latest version)
- Set node labels (accelerator)
- Ingress Controller (e.g. NGINX)
- Storage Provisioner (e.g. aws-ebs)
- Image Pull Secret for nvcr.io
- TLS Secret/certificate

### Hardware setup
GPU Node(s) with Ubuntu 18.04 or latest

Minimum:
- 4 GB system RAM
- 2.5 GB of GPU RAM
- 6 core CPU
- 1 NVIDIA GPU
  + Discrete GPU: NVIDIA Volta, Turing, or Ampere GPU architecture
- 12 GB of HDD/SSD space

Recommended:
- 32 GB system RAM
- 32 GB of GPU RAM
- 8 core CPU
- 1 NVIDIA GPU
  + Discrete GPU: Volta, Turing, Ampere architecture
- 16 GB of SSD space

### Kubernetes install
On AWS we suggest using EKS. Please refer to:
https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-eks-setup.html

For Bare Metal, please refer to Kubespray's README.md: https://github.com/kubernetes-sigs/kubespray
Last tested version at time of this writing: v2.18.0.

Common mistakes:
- Not starting form a fresh Ubuntu system (exempt of "nouveau" driver, NVIDIA driver and NVIDIA docker2)
- Not setting up ssh passwordless to access cluster nodes
- Too many domains listed under search in your nodes' /etc/resolv.conf
- Not disabling swap on your cluster nodes
- Not enabling docker_storage_options: -s overlay2
- Not enabling helm_enabled: true
- After installation, not making a user-readable copy of /etc/kubernetes/admin.conf to ~/.kube/config

### NVIDIA GPU Operator
On AWS EKS, use the NVIDIA device plugin described from above link.
Other systems can deploy newer gpu-operator with:
```
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update
helm install gpu-operator nvidia/gpu-operator --set driver.repository=nvcr.io/nvidia --set driver.version="510.47.03"
```
Note that if GPU Operator was previously uninstalled, you might need to run the followin before a new install:
```
kubectl delete crd clusterpolicies.nvidia.com
```
One can wait a few minutes and check all GPU related pods are in good health with:
```
kubectl get pods -A
```
If the GPU pods are failing, check once more that no "nouveau" or NVIDIA drivers are enabled on the nodes.
One can also make sure nodes are schedulable with:
```
kubectl get nodes -o name | xargs kubectl uncordon
```

### Set node labels
List nodes and labels with:
```
kubectl get nodes --show-labels
```
Then, set node label with (example):
```
kubectl label nodes node1 accelerator=rtx3090
```

### NGINX Ingress Controller
```
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install ingress-nginx ingress-nginx/ingress-nginx
```

### Storage Provisioner
On AWS EKS, one might look at aws-ebs or aws-efs.
Below is an example with local NFS (requires a local NFS server). One must replace the NFS server IP and exported path.
```
helm repo add nfs-subdir-external-provisioner https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/
helm install nfs-subdir-external-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
  --set nfs.server=172.17.171.248 \
  --set nfs.path=/srv/nfs/kubedata \
  --set storageClass.onDelete=retain \
  --set storageClass.pathPattern="\${.PVC.namespace}-\${.PVC.name}"
```

### Image Pull Secret for nvcr.io
In this example, one must set his ngc-api-key, ngc-email and deployment namespace.
```
kubectl create secret docker-registry 'imagepullsecret' --docker-server='nvcr.io' --docker-username='$oauthtoken' --docker-password='ngc-api-key' --docker-email='ngc-email' --namespace='default'
```
Where:
- ngc-api-key can be obtained from https://catalog.ngc.nvidia.com/ after signing in, by selecting Setup Menu :: API KEY :: Get API KEY :: Generate API Key
- ngc-email is the email one used to sign in above
- namespace is the Kubernetes namespace one uses for deployment, or “default”


### BCP Cluster Secret to Submit job in BCP Cluster
When ngcRunner is set to True, jobs are running in the Nvidia BCP cluster. In this example, one must set his ngc-api-key that can submit job to BCP cluster.
```
kubectl create secret docker-registry 'bcpclustersecret' --docker-server='nvcr.io' --docker-username='$oauthtoken' --docker-password='ngc-api-key' --docker-email='ngc-email' --namespace='default'
```


### Admin secret to update job status and logs
In this example, one must set admin ngc-api-key that can callback from job containers to update status and logs.
```
kubectl create secret docker-registry 'adminclustersecret' --docker-server='nvcr.io' --docker-username='$oauthtoken' --docker-password='ngc-api-key' --docker-email='ngc-email' --namespace='default'
```


### TLS Secret/certificate
To create your certificate, please refer to: [tls-certificates](https://kubernetes.github.io/ingress-nginx/examples/PREREQUISITES/#tls-certificates)

The following is an example for creating a secret from your certificate:
```
kubectl create secret tls tls-secret --key tls.key --cert tls.crt --namespace default
```

## Deployment
One must update the chart's values.yaml before deployment.
If your cluster has no default Storage Provisioner, you would have to set StorageClassName to [for example] local-path.
For HTTPS, you would have to set your host and tlsSecret.
```
helm install tao-api chart/ --namespace default
```

## <a name='License'></a>License
This project is licensed under the [Apache-2.0](./LICENSE) License.

