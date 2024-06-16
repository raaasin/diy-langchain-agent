import io
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

file="image.png"
image_data = open(file, "rb").read()
image_bytes = io.BytesIO(image_data)

safe = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]


picture = {
        'mime_type': 'image/png',
        'data': image_bytes.getvalue()  
        }

response = model.generate_content(
                ["""OCR all the contents and reply like this calories(if not visible then None):,nutrients(if not visible then None):[],ingredients(if not visible then None)[],general_product_name(if not visible then None):,ignore daily percentages etc, i just want numbers based on the nutritional composition per 100 grams or 100 milliliters of the product, rather than per serving and reply in a json format, 
                 
                 example (you can add more nutrients or ingredients based on the image):
                 {"calories": 160, "nutrients": ["total_fat":8,"saturated_fat": 3, "trans_fat": 0,"cholestrol":0,"Sodium":4,etc],"general_product_name": Chips}
                 
                  """, picture],
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0
                ),safety_settings=safe
            )
response.resolve()


print(response)