# Import necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
load_dotenv()
import os
import json

gemini_api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0,google_api_key=gemini_api_key)

# Define the tool for calculating Health Star Rating
@tool
def calculate_health_star_rating(nutritional_info: str) -> str:
    """
    Calculates the Health Star Rating based on the provided nutritional information.
    The input should be a JSON string with the following format:
    {
        "calories": 160,
        "nutrients": [
            {"name": "Total Fat", "amount": 8, "unit": "g"},
            {"name": "Saturated Fat", "amount": 3, "unit": "g"},
            ...
        ],
        "general_product_name": None
    }
    """
    data = json.loads(nutritional_info)
    
    # Extract the required nutrients
    calories = data.get("calories", 0)
    nutrients = {nutrient["name"]: nutrient["amount"] for nutrient in data["nutrients"]}
    
    # Initialize baseline points and modifying points
    baseline_points = 0
    modifying_points = 0
    
    # Calculate baseline points (example values for illustration)
    baseline_points += int(calories / 100)
    baseline_points += int(nutrients.get("Total Fat", 0))
    baseline_points += int(nutrients.get("Saturated Fat", 0))
    baseline_points += int(nutrients.get("Total Sugars", 0))
    baseline_points += int(nutrients.get("Sodium", 0) / 100)
    
    # Calculate modifying points (example values for illustration)
    modifying_points += int(nutrients.get("Protein", 0) / 10)
    modifying_points += int(nutrients.get("Dietary Fiber", 0) / 5)
    
    # Calculate final score
    final_score = baseline_points - modifying_points
    
    # Determine Health Star Rating
    if final_score <= -6:
        hsr = 5
    elif final_score <= -2:
        hsr = 4.5
    elif final_score <= 0:
        hsr = 4
    elif final_score <= 2:
        hsr = 3.5
    elif final_score <= 4:
        hsr = 3
    elif final_score <= 6:
        hsr = 2.5
    elif final_score <= 8:
        hsr = 2
    elif final_score <= 10:
        hsr = 1.5
    elif final_score <= 12:
        hsr = 1
    else:
        hsr = 0.5

    return f"The Health Star Rating is {hsr} stars."

# Example invocation
nutritional_info_example = '{"calories": 160, "nutrients": [{"name": "Total Fat", "amount": 8, "unit": "g"}, {"name": "Saturated Fat", "amount": 3, "unit": "g"}, {"name": "Total Sugars", "amount": 15, "unit": "g"}, {"name": "Sodium", "amount": 60, "unit": "mg"}, {"name": "Protein", "amount": 3, "unit": "g"}, {"name": "Dietary Fiber", "amount": 3, "unit": "g"}], "general_product_name": null}'
calculate_health_star_rating.invoke(nutritional_info_example)

# Create the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very powerful assistant, but you don't know current events."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# Bind tools to LLM
tools = [calculate_health_star_rating]
llm_with_tools = llm.bind_tools(tools)

# Create the agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


test_cases = [
    {
        "input": '{"calories": 500, "nutrients": [{"name": "Total Fat", "amount": 20, "unit": "g"}, {"name": "Saturated Fat", "amount": 10, "unit": "g"}, {"name": "Total Sugars", "amount": 25, "unit": "g"}, {"name": "Sodium", "amount": 300, "unit": "mg"}, {"name": "Protein", "amount": 5, "unit": "g"}, {"name": "Dietary Fiber", "amount": 2, "unit": "g"}], "general_product_name": null}'
    },
    {
        "input": '{"calories": 200, "nutrients": [{"name": "Total Fat", "amount": 5, "unit": "g"}, {"name": "Saturated Fat", "amount": 1, "unit": "g"}, {"name": "Total Sugars", "amount": 5, "unit": "g"}, {"name": "Sodium", "amount": 50, "unit": "mg"}, {"name": "Protein", "amount": 10, "unit": "g"}, {"name": "Dietary Fiber", "amount": 10, "unit": "g"}], "general_product_name": null}'
    },
    {
        "input": '{"calories": 300, "nutrients": [{"name": "Total Fat", "amount": 15, "unit": "g"}, {"name": "Saturated Fat", "amount": 5, "unit": "g"}, {"name": "Total Sugars", "amount": 10, "unit": "g"}, {"name": "Sodium", "amount": 700, "unit": "mg"}, {"name": "Protein", "amount": 15, "unit": "g"}, {"name": "Dietary Fiber", "amount": 3, "unit": "g"}], "general_product_name": null}'
    },
    {
        "input": '{"calories": 150, "nutrients": [{"name": "Total Fat", "amount": 3, "unit": "g"}, {"name": "Saturated Fat", "amount": 1, "unit": "g"}, {"name": "Total Sugars", "amount": 10, "unit": "g"}, {"name": "Sodium", "amount": 100, "unit": "mg"}, {"name": "Protein", "amount": 5, "unit": "g"}, {"name": "Dietary Fiber", "amount": 5, "unit": "g"}], "general_product_name": null}'
    },
    {
        "input": '{"calories": 250, "nutrients": [{"name": "Total Fat", "amount": 10, "unit": "g"}, {"name": "Saturated Fat", "amount": 2, "unit": "g"}, {"name": "Total Sugars", "amount": 5, "unit": "g"}, {"name": "Sodium", "amount": 150, "unit": "mg"}, {"name": "Protein", "amount": 20, "unit": "g"}, {"name": "Dietary Fiber", "amount": 2, "unit": "g"}], "general_product_name": null}'
    }
]


def run_test_cases():
    results = []
    for i, test_case in enumerate(test_cases):
        result = list(agent_executor.stream(test_case))
        results.append({"Test Case": i+1, "Result": result})
    return results

# Run the test cases and print the results
test_results = run_test_cases()
for test in test_results:
    print(f"Test Case {test['Test Case']}: {test['Result']}")

# If you want to see the output in a more readable format
import pprint
pprint.pprint(test_results)
