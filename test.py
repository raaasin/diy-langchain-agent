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
                ["""OCR all the contents and reply like this calories:,nutrients:[],general_product_name(if visible):,ignore daily percentages etc, i just want numbers and reply in a json format, 
                 
                 example:
                 {"calories": 160, "nutrients": [{"name": "Total Fat", "amount": 8, "unit": "g"}, {"name": "Saturated Fat", "amount": 3, "unit": "g"}, {"name": "Trans Fat", "amount": 0, "unit": "g"}, {"name": "Cholesterol", "amount": 0, "unit": "mg"}, {"name": "Sodium", "amount": 60, "unit": "mg"}, {"name": "Total Carbohydrate", "amount": 21, "unit": "g"}, {"name": "Dietary Fiber", "amount": 3, "unit": "g"}, {"name": "Total Sugars", "amount": 15, "unit": "g"}, {"name": "Includes Added Sugars", "amount": 5, "unit": "g"}, {"name": "Protein", "amount": 3, "unit": "g"}, {"name": "Vitamin D", "amount": 5, "unit": "mcg"}, {"name": "Calcium", "amount": 20, "unit": "mg"}, {"name": "Iron", "amount": 1, "unit": "mg"}, {"name": "Potassium", "amount": 230, "unit": "mg"}], "general_product_name": None}
                 
                  """, picture],
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0
                ),safety_settings=safe
            )
response.resolve()
response = response.text.replace("```", "")
response = response.replace("null", "None")
response = response.replace("json", "")
response_data = eval(response)

print(response)