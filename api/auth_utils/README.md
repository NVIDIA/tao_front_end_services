
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
import jwt
import uuid

def getUser(token):
  user = None
  payload = {}
  try:
    jwks_client = jwt.PyJWKClient("https://authn.nvidia.com/pubJWKS")
    signing_key = jwks_client.get_signing_key_from_jwt(token)
    payload = jwt.decode(
      token,
      signing_key.key,
      algorithms = ["RS256"]
    )
    user = uuid.uuid5(uuid.UUID(int=0), payload.get('sub'))
  except Exception as e:
    print('Token error: ' + str(e))
  return user
```

## Example of API call with Authorization Bearer Token

```python
import requests

def apiPost(url, data, token):
  r = requests.post(url=endpoint, json=data, headers={'Authorization': 'Bearer ' + token})
  assert r.status_code is 201
```
## Example URL

```
https://10.111.60.42:30351/api/v1/user/f6934afa-2b33-47e5-90d6-606eea0b9f96/model
```
