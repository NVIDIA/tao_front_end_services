
# NGC Authentication

## Example to get JWT token from NGC_API_KEY

```python
import requests

def getToken(key):
  r = requests.get('https://authn.nvidia.com/token', headers={'Accept': 'application/json', 'Authorization': 'ApiKey ' + key})
  assert r.status_code is 200
  return r.json().get('token')
```

## Example to get user ID from JWT

```python
import uuid

def getUser(token):
  headers = {'Accept': 'application/json', 'Authorization': 'Bearer ' + token}
  r = requests.get('https://api.ngc.nvidia.com/v2/users/me', headers=headers)
  assert r.status_code is 200
  ngc_user_id = r.json().get('user', {}).get('id')
  user_id = str(uuid.uuid5(uuid.UUID(int=0), str(ngc_user_id)))
  return user_id
```

## Example of API call with Authorization Bearer Token

```python
import requests

def apiPost(url, data, token):
  r = requests.post(url=endpoint, json=data, headers={'Authorization': 'Bearer ' + token})
  assert r.status_code is 201
```

## Example of API call with SID or SSID Cookie from NGC UI

```python
import requests

def apiPost(url, data, token):
  r = requests.post(url=endpoint, json=data, headers={'Cookie': 'SSID=' + token})
  assert r.status_code is 201
```

## Example URL

```
https://10.111.60.42:30351/api/v1/users/f6934afa-2b33-47e5-90d6-606eea0b9f96/experiments
```
