import requests

url = "https://api.play.ht/api/v1/getClonedVoices"

response = requests.get(url)

print(response.text)