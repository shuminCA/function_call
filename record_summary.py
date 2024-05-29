import base64
import anthropic
import httpx
import os
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    image_media_type = "image/jpeg"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    message = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')).beta.tools.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        tools=[
            {
                "name": "record_summary",
                "description": "Record summary an image into well-structured JSON.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key_colors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "r": {
                                        "type": "number",
                                        "description": "red value [0.0, 1.0]",
                                    },
                                    "g": {
                                        "type": "number",
                                        "description": "green value [0.0, 1.0]",
                                    },
                                    "b": {
                                        "type": "number",
                                        "description": "blue value [0.0, 1.0]",
                                    },
                                    "name": {
                                        "type": "string",
                                        "description": "Human-readable color name in snake_case, e.g. \"olive_green\" or \"turquoise\""
                                    },
                                },
                                "required": ["r", "g", "b", "name"],
                            },
                            "description": "Key colors in the image. Limit to less then four.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Image description. One to two sentences max.",
                        },
                        "estimated_year": {
                            "type": "integer",
                            "description": "Estimated year that the images was taken, if it a photo. Only set this if the image appears to be non-fictional. Rough estimates are okay!",
                        },
                    },
                    "required": ["key_colors", "description"],
                },
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": "Use `record_summary` to describe this image."},
                ],
            }
        ],
    )
    print(message)