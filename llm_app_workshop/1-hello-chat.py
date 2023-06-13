import ray
import requests, json
from starlette.requests import Request
from typing import Dict

from ray import serve

@serve.deployment
class Chat:
    def __init__(self, msg: str):
        self._msg = msg # initial state

    async def __call__(self, request: Request) -> Dict:
        data = await request.json()
        data = json.loads(data)
        return {"result": self.get_response(data['user_input']) }
    
    def get_response(self, message: str) -> str:
        return self._msg + message

entrypoint = Chat.bind(msg="Yes... ")

