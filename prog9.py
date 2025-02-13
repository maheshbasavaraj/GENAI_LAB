# Import libraries
import wikipediaapi
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI  # Replace with Cohere or another LLM if preferred


# Step 1: Define Pydantic schema for output
class InstitutionDetails(BaseModel):
    founder: str
    founded: str
    branches: List[str]
    employees: Optional[int]
    summary: str


# Step 2: Create a custom parser
parser = PydanticOutputParser(pydantic_object=InstitutionDetails)


# Step 3: Fetch data from Wikipedia
def fetch_wikipedia_data(institution_name: str) -> str:
    wiki = wikipediaapi.Wikipedia("en")
    page = wiki.page(institution_name)
    return page.text if page.exists() else None


# Step 4: Define prompt template for LLM
template = """
Extract the following details about {institution} from the text below:
- Founder
- Year founded
- Current branches (as a list)
- Number of employees (if available)
- A 4-line summary

Text:
{text}

Format the output as JSON using this schema:
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["institution", "text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Step 5: Initialize LLM and chain
llm = OpenAI(temperature=0)  # Replace with your LLM API key
chain = LLMChain(llm=llm, prompt=prompt)

# Step 6: Main program
institution = input("Enter institution name (e.g., 'Harvard University'): ").strip()
text = fetch_wikipedia_data(institution)

if not text:
    print("Error: Institution not found on Wikipedia.")
else:
    try:
        output = chain.run(institution=institution, text=text)
        parsed_data = parser.parse(output)
        print("\n=== Institution Details ===")
        print(f"Founder: {parsed_data.founder}")
        print(f"Founded: {parsed_data.founded}")
        print(f"Branches: {', '.join(parsed_data.branches)}")
        print(f"Employees: {parsed_data.employees or 'Not available'}")
        print(f"Summary:\n{parsed_data.summary}")
    except ValidationError as e:
        print(f"Validation Error: {e}")
