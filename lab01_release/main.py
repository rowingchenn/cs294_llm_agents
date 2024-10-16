from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import math
import json
import re

def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # This function takes in a restaurant name and returns the reviews for that restaurant. 
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    # The "data fetch agent" should have access to this function signature, and it should be able to suggest this as a function call. 
    # Example:
    # > fetch_restaurant_data("Applebee's")
    # {"Applebee's": ["The food at Applebee's was average, with nothing particularly standing out.", ...]}
    restaurant_reviews = {}
    restaurant_name = re.sub(r"[\s\-]", "", restaurant_name.lower())
    with open("restaurant-data.txt", "r") as f:
        lines = f.readlines()
        
    for line in lines:
        name = line.split(".", 1)[0]
        name = re.sub(r"[\s\-]", "", name.lower())
        if name != restaurant_name:
            continue
        review = line.split(".", 1)[1]
        if name not in restaurant_reviews:
            restaurant_reviews[name] = []
        restaurant_reviews[name].append(review)
        
    return restaurant_reviews
    

def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    # TODO
    # This function takes in a restaurant name, a list of food scores from 1-5, and a list of customer service scores from 1-5
    # The output should be a score between 0 and 10, which is computed as the following:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    # The above formula is a geometric mean of the scores, which penalizes food quality more than customer service. 
    # Example:
    # > calculate_overall_score("Applebee's", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    # {"Applebee's": 5.04}
    # NOTE: be sure to round the score to 2 decimal places.
    if len(food_scores) != len(customer_service_scores):
        raise ValueError("The number of food scores and customer service scores must be the same.")
    
    n = len(food_scores)
    score = 0
    
    for i in range(n):
        score += math.sqrt(food_scores[i]**2 * customer_service_scores[i])
        
    score *= 1/(n * math.sqrt(125)) * 10
    
    return {restaurant_name: round(score, 3)+1e-8}


# TODO: feel free to write as many additional functions as you'd like.

# Do not modify the signature of the "main" function.
def main(user_query: str):
    data_fetch_prompt = "You can help users find information about curtain restaurants in the database. " 
    # example LLM config for the entrypoint agent
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    # the main entrypoint/supervisor agent
    data_fetch_agent = ConversableAgent(
        "Data Fetch Agent", 
        system_message=data_fetch_prompt, 
        llm_config=llm_config,
        max_consecutive_auto_reply=1,
        human_input_mode="NEVER",  # Never ask for human input.
        silent = True
    )
    entrypoint_agent = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
        silent = True
    )
    
    review_analysis_prompt = """
    Analyze the restaurant review to extract two scores: food_score and customer_service_score, both from 1 to 5.

    Look for two specific adjectives in the review:

        •	Food: adjectives that describe the food quality.
        •	Customer service: adjectives that describe the service quality.

    Use the following scoring system based on the adjectives:

        •	1/5: awful, horrible, disgusting
        •	2/5: bad, unpleasant, offensive
        •	3/5: average, uninspiring, forgettable
        •	4/5: good, enjoyable, satisfying
        •	5/5: awesome, incredible, amazing

    Output format:

        •	food_score: (score)
        •	customer_service_score: (score)

    Example:

    Review:

        “The food was average, but the customer service was unpleasant.”

    Output:

        •	food_score: 3
        •	customer_service_score: 2

    Now, analyze this review:
    """

    review_analysis_agent = ConversableAgent(
        name="Review Analysis Agent",
        system_message=review_analysis_prompt,
        llm_config=llm_config,
        max_consecutive_auto_reply=1,
        human_input_mode="NEVER",
        silent = True
    )
    
    scoring_prompt = """
            Calculate the overall score for the restaurant based on the food and customer service scores.
        """

    scoring_agent = ConversableAgent(
        name="Scoring Agent",
        system_message=scoring_prompt,
        llm_config=llm_config,
        max_consecutive_auto_reply=1,
        human_input_mode="NEVER",
        silent = True
)
    
    data_fetch_agent.register_for_llm(name="fetch_reviews_for_restaurant", description="Fetches the reviews for a specific restaurant.")(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="fetch_reviews_for_restaurant")(fetch_restaurant_data)
    
    scoring_agent.register_for_llm(name="calculate_overall_score", description="Calculate the overall score based on the review analysing results.")(calculate_overall_score)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)    
    
    data_fetch_result = entrypoint_agent.initiate_chat(data_fetch_agent, message=user_query)
    restaurant_name = json.loads((data_fetch_result.chat_history[1]
                   .get("tool_calls", [{}])[0]
                   .get("function", {})
                   .get("arguments", {})))
    name = re.sub(r"[\s\-]", "", restaurant_name["restaurant_name"].lower())
    reviews = json.loads(data_fetch_result.chat_history[-1]["content"])
    reviews = reviews[name]
    n = len(reviews)
    total_score = 0

    for review in reviews:
        review_analysis = entrypoint_agent.initiate_chat(review_analysis_agent, message=review)
        result = review_analysis.chat_history[1]["content"]
        score_result = entrypoint_agent.initiate_chat(scoring_agent, message=name + ": " + result, max_turns=2)
        score = json.loads(score_result.chat_history[-1]["content"]).get(name)
        total_score += score
        
    
    print(total_score/n)
    return 
    # TODO
    # Create more agents here. 
    
    # TODO
    # Fill in the argument to `initiate_chats` below, calling the correct agents sequentially.
    # If you decide to use another conversation pattern, feel free to disregard this code.
    
    # Uncomment once you initiate the chat with at least one agent.
    # result = entrypoint_agent.initiate_chats([{}])
    
# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])