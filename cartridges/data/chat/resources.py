from cartridges.data.resources import Resource

class ChatHistoryResource(Resource):

    class Config(Resource.Config):
        path: str

    def __init__(self, config: Config):
        self.config = config
    
    async def setup(self):
        

    def read(self):
        with open(self.path, "r") as f:
            return f.read()