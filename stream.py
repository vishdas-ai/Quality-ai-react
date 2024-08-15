import json
import logging
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
DATASET_NAME = os.getenv("DATASET_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

client = bigquery.Client()

def get_table_schema_and_description(dataset_name):
    dataset_ref = client.dataset(dataset_name)
    table_names = ['qnps_embedding', 'warranty_embedding']
    table_schemas = {}

    def get_fields(schema):
        return [(field.name, field.field_type, field.mode, field.description) for field in schema]

    for table_name in table_names:
        table_ref = dataset_ref.table(table_name)
        table = client.get_table(table_ref)
        table_schemas[table.table_id] = {
            'description': table.description,
            'schema': get_fields(table.schema)
        }
    
    return table_schemas

def determine_search_method(query):
    table_schema = get_table_schema_and_description(DATASET_NAME)
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    
    prompt = f"""
    Given the following user question and table schemas, determine which table to use and whether to use vector search or text2sql. Analyze the question carefully and consider the available columns and their descriptions in each table.

    User question: {query}
    Table schemas: {table_schema}

    Table schemas:
    1. qnps_embedding:
       - Description: Stores customer feedback and sentiment analysis data on various aspects of vehicles.
       - Key columns: verbatim (customer feedback text), verbatim_embedding (vector representation of feedback), sentiment, polarity, ccc_code (Customer Concern Category), vfg_code (Vehicle Functional Group), function_code

    2. warranty_embedding:
       - Description: Stores comprehensive data related to warranty claims, including claim details, vehicle information, repair process, and associated costs.
       - Key columns: combined_issue_details (customer and technician descriptions), combined_issue_details_embedding (vector representation of issue details), dtc_code (Diagnostic Trouble Code), cust_conc_cd (Customer Concern Code), lbr_cost, mtrl_cost, tot_cost_gross

    Respond with the following information in JSON format:
    1. table_name: The name of the table to use (either "qnps_embedding" or "warranty_embedding")
    2. search_method: Either "vector_search" or "text2sql"
    3. reason: A detailed explanation for your choice, referencing specific columns and aspects of the question

    Guidelines for selection:
    - Use vector_search if:
      a) The question asks for similar or related items
      b) The question requires semantic understanding or natural language processing
      c) The question is about finding patterns or trends in customer feedback or issue descriptions
      d) The relevant information is likely contained within text fields (verbatim or combined_issue_details)

    - Use text2sql if:
      a) The question can be answered directly using specific columns in the table
      b) The question involves precise numerical calculations or aggregations
      c) The question requires filtering or grouping based on specific criteria
      d) The relevant information is stored in structured fields rather than text descriptions

    Consider the nature of the data in each table:
    - qnps_embedding is better for questions about customer sentiment, feedback trends.
    - warranty_embedding is better for questions about repair costs, frequent issues, specific feature-related  or specific diagnostic codes

    Provide only the JSON object without any additional formatting or explanation.
    """
    
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "top_p": 0.8},
    )

    # Clean up the response text
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()

    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse response as JSON. Raw response: {cleaned_response}")
        raise ValueError("Invalid JSON response from determine_search_method function")
    
async def vector_search(query_text, table_name):
    table_id = f"{PROJECT_ID}.{DATASET_NAME}.{table_name}"
    try:
        if table_name == "warranty_embedding":
            query = f"""
            SELECT base.dtc_code, base.cust_conc_cd, base.combined_issue_details
            FROM VECTOR_SEARCH(
                TABLE `{table_id}`, 'combined_issue_details_embedding',
                (SELECT ml_generate_embedding_result, content AS query
                 FROM ML.GENERATE_EMBEDDING(
                     MODEL `{PROJECT_ID}.{DATASET_NAME}.textembedding`,
                     (SELECT '{query_text}' AS content))
                ),
                top_k => 5, options => '{{"fraction_lists_to_search": 0.01}}')
            """
        elif table_name == "qnps_embedding":
            query = f"""
            SELECT base.ccc_code, base.verbatim
            FROM VECTOR_SEARCH(
                TABLE `{table_id}`, 'verbatim_embedding',
                (SELECT ml_generate_embedding_result, content AS query
                 FROM ML.GENERATE_EMBEDDING(
                     MODEL `{PROJECT_ID}.{DATASET_NAME}.textembedding`,
                     (SELECT '{query_text}' AS content))
                ),
                top_k => 5, options => '{{"fraction_lists_to_search": 0.01}}')
            """
        else:
            raise ValueError(f"Unsupported table_name: {table_name}")

        query_job = await asyncio.to_thread(client.query, query)
        results = await asyncio.to_thread(query_job.result)
        logger.info(f"Performed vector search on table {table_id}")
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error performing vector search on table {table_id}: {str(e)}")
        raise
    
async def generate_and_execute_sql(query, table_name):
    table_schema = get_table_schema_and_description(DATASET_NAME)
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    
    sql_query_prompt = f"""
    Given the table schema: {table_schema[table_name]}, generate a SQL query to answer the user question: {query}
    Use the following table id: {PROJECT_ID}.{DATASET_NAME}.{table_name}
    
    Provide only the SQL query without any additional formatting or explanation.
    """
    
    sql_query_response = model.generate_content(
        sql_query_prompt,
        generation_config={"temperature": 0.2, "top_p": 0.8},
    )
    
    sql_query = sql_query_response.text.strip()
    cleaned_query = (
            sql_query
            .replace("\\n", " ")
            .replace("\n", " ")
            .replace("\\", "")
            .replace("```sql", "")
            .replace("```", "")
            .strip()
        )
    
    try:
        query_job = client.query(cleaned_query)
        results = query_job.result()
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        raise

async def generate_natural_language_answer(query, results):
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    
    prompt = f"""
    Given the following user question and query results, provide a concise natural language answer:

    User question: {query}
    Query results: {results}

    Please summarize the results and directly answer the user's question in a clear and concise manner.Dont miss any information from the results..
    """
    
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "top_p": 0.8},
    )
    
    return response.text.strip()

async def process_query(query):
    try:
        # Step 1: Determine the search method and table
        search_info = determine_search_method(query)
        logger.info(f"Search method determined: {search_info}")
        
        # Step 2: Perform the search
        if search_info['search_method'] == 'vector_search':
            results = await vector_search(query, search_info['table_name'])
        elif search_info['search_method'] == 'text2sql':
            results = await generate_and_execute_sql(query, search_info['table_name'])
        else:
            raise ValueError(f"Invalid search method: {search_info['search_method']}")
        
        # Step 3: Generate natural language answer
        answer = await generate_natural_language_answer(query, results)
        
        return {
            'query': query,
            'search_method': search_info['search_method'],
            'table_name': search_info['table_name'],
            'results': results,
            'answer': answer
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {'error': str(e)}

async def main():
    query = "How does the distribution of claim types vary across different regions?"
    result = await process_query(query)
    if 'error' in result:
        print(f"An error occurred: {result['error']}")
    else:
        print(f"Query: {result['query']}")
        print(f"Search method: {result['search_method']}")
        print(f"Table used: {result['table_name']}")
        print(f"Results: {result['results']}")
        print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())

