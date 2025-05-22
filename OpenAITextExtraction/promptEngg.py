from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
# from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

# Define the output structure
class PersonInfo(BaseModel):
    name: str = Field(description="The person's name")
    age: str = Field(description="The person's age")
    salary: str = Field(description="The person's total compensation (salary + stocks + bonuses)")

# Create a parser
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# Create a system message with multiple examples
system_message = """
Extract information about a person from the text and format it as structured data.
Follow these rules:
1. Extract the person's name
2. Extract the person's age
3. For salary, calculate the TOTAL compensation by adding all monetary values (salary, stocks, bonuses, etc.)

EXAMPLES:

Example 1:
Input: "Soumya is 40 years old and gets 15k salary"
Output: {{"name": "Soumya", "age": "40", "salary": "15k"}}

Example 2:
Input: "Mark is 35 and receives 20k in salary and 10k in stock options"
Output: {{"name": "Mark", "age": "35", "salary": "30k"}}

Example 3:
Input: "Priya, who is 28, earns a base salary of 18k with a quarterly bonus of 5k"
Output: {{"name": "Priya", "age": "28", "salary": "23k"}}

Example 4:
Input: "Alex is 42 years old with compensation of 25k salary, 15k RSUs, and 10k bonus"
Output: {{"name": "Alex", "age": "42", "salary": "50k"}}

Now extract information from the following text:
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "{input_text}")
])

print(prompt)

# Set up the model and chain
# model = ChatOpenAI(temperature=0)
#
# # Option 1: Using the parser directly
# extraction_chain = prompt | model | parser
#
# # Option 2: For more reliable extraction with function calling
# from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
#
# function_chain = (
#     prompt
#     | model.bind_functions(functions=[PersonInfo], function_call="PersonInfo")
#     | PydanticAttrOutputFunctionsParser(pydantic_schema=PersonInfo)
# )
#
# # Example usage
# result = extraction_chain.invoke({"input_text": "Raj is 32 years old and gets 22k salary plus 8k in stocks"})
# print(result)  # Should return: {"name": "Raj", "age": "32", "salary": "30k"}