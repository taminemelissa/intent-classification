class BaseModel:
    def __init__(self, device: str = None):
        self.name = type(self).__name__
        self._device = device
        self.tokenizer = None
        self.model = None
    
    def device(self):
        return self._device
    
