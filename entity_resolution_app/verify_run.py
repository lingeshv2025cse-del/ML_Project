import os
import sys

sys.path.insert(0, os.getcwd())

from app import app

client = app.test_client()
response = client.get('/run-sample')
print('STATUS', response.status_code)
print('CONTENT_TYPE', response.mimetype)
print(response.data.decode('utf-8')[:500])
print('MATCHES_EXISTS', os.path.exists('output/matches.csv'))
print('UNMATCHED_EXISTS', os.path.exists('output/unmatched.csv'))
