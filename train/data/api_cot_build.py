import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
from PIL import Image
import io
from typing import List, Dict, Any, Optional

class ImageProcessor:
    """
    A class for processing images for API requests.
    """
    
    @staticmethod
    def resize_image(image_path: str, size: tuple) -> Image.Image:
        """
        Resize an image to the specified dimensions.
        
        Args:
            image_path (str): Path to the image file
            size (tuple): Target dimensions (width, height)
            
        Returns:
            Image.Image: Resized image
        """
        image = Image.open(image_path)
        resized_image = image.resize(size)
        return resized_image

    @staticmethod
    def encode_image(image: Image.Image) -> tuple:
        """
        Encode an image to base64 string.
        
        Args:
            image (Image.Image): PIL Image object
            
        Returns:
            tuple: (image_byte_array, base64_string)
        """
        image_byte_array = io.BytesIO()
        image.save(image_byte_array, format='PNG')
        image_byte_array = image_byte_array.getvalue()
        return image_byte_array, base64.b64encode(image_byte_array).decode('utf-8')

class WsParam:
    """
    WebSocket parameter handler for API authentication.
    """
    
    def __init__(self, app_id: str, api_key: str, api_secret: str, image_understanding_url: str):
        """
        Initialize WebSocket parameters.
        
        Args:
            app_id (str): Application ID
            api_key (str): API Key
            api_secret (str): API Secret
            image_understanding_url (str): API endpoint URL
        """
        self.APPID = app_id
        self.APIKey = api_key
        self.APISecret = api_secret
        self.host = urlparse(image_understanding_url).netloc
        self.path = urlparse(image_understanding_url).path
        self.ImageUnderstanding_url = image_understanding_url

    def create_url(self) -> str:
        """
        Generate authenticated URL for WebSocket connection.
        
        Returns:
            str: Authenticated WebSocket URL
        """
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        signature_sha = hmac.new(
            self.APISecret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        return self.ImageUnderstanding_url + '?' + urlencode(v)

class APIClient:
    """
    Client for handling API communication and image understanding.
    """
    
    def __init__(self, app_id: str, api_key: str, api_secret: str, image_understanding_url: str):
        """
        Initialize API client.
        
        Args:
            app_id (str): Application ID
            api_key (str): API Key
            api_secret (str): API Secret
            image_understanding_url (str): API endpoint URL
        """
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.image_understanding_url = image_understanding_url
        self.answer = ""

    def on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket errors."""
        print(f"Error: {error}")

    def on_close(self, ws: websocket.WebSocketApp, *args) -> None:
        """Handle WebSocket connection closure."""
        pass

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection opening."""
        thread.start_new_thread(self.run, (ws,))

    def run(self, ws: websocket.WebSocketApp, *args) -> None:
        """Send initial request after connection."""
        data = json.dumps(self.gen_params(ws.appid, ws.question))
        ws.send(data)

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages."""
        data = json.loads(message)
        code = data['header']['code']
        
        if code != 0:
            print(f'Request error: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            print(content, end="")
            self.answer += content
            
            if status == 2:
                print("Usage:", data["payload"]['usage'])
                ws.close()

    def gen_params(self, appid: str, question: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate request parameters.
        
        Args:
            appid (str): Application ID
            question (List[Dict[str, str]]): Question context
            
        Returns:
            Dict[str, Any]: Request parameters
        """
        return {
            "header": {"app_id": appid},
            "parameter": {
                "chat": {
                    "domain": "image",
                    "temperature": 0.5,
                    "top_k": 4,
                    "max_tokens": 2028,
                    "auditing": "default"
                }
            },
            "payload": {"message": {"text": question}}
        }

    def process_image(self, image_path: str, size: tuple = (1000, 1000)) -> tuple:
        """
        Process image for API request.
        
        Args:
            image_path (str): Path to image file
            size (tuple): Target dimensions
            
        Returns:
            tuple: (image_data, base64_image)
        """
        resized_image = ImageProcessor.resize_image(image_path, size)
        return ImageProcessor.encode_image(resized_image)

    def get_text(self, role: str, content: str) -> List[Dict[str, str]]:
        """
        Format text for API request.
        
        Args:
            role (str): Role of the message sender
            content (str): Message content
            
        Returns:
            List[Dict[str, str]]: Formatted message list
        """
        return [{"role": role, "content": content}]

    def check_length(self, text: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Check and adjust text length to meet API requirements.
        
        Args:
            text (List[Dict[str, str]]): Message list
            
        Returns:
            List[Dict[str, str]]: Adjusted message list
        """
        while len(json.dumps(text[1:])) > 8000:
            del text[1]
        return text

    def main(self, image_path: str, question: str) -> None:
        """
        Main execution function.
        
        Args:
            image_path (str): Path to image file
            question (str): Question to ask about the image
        """
        imagedata, base64_image = self.process_image(image_path)
        ws_param = WsParam(self.app_id, self.api_key, self.api_secret, self.image_understanding_url)
        websocket.enableTrace(False)
        ws_url = ws_param.create_url()
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        ws.appid = self.app_id
        ws.imagedata = imagedata
        ws.question = self.check_length(self.get_text("user", question))
        
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

def main():
    """
    Main execution function.
    """
    # Replace with your actual API credentials
    app_id = "YOUR_APP_ID"
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    image_understanding_url = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"
    
    client = APIClient(app_id, api_key, api_secret, image_understanding_url)
    image_path = "images/sample.png"
    question = "Describe this image in detail."
    
    print("Answer:", end="")
    client.main(image_path, question)

if __name__ == '__main__':
    main()


