#import os
#from dotenv import load_dotenv
#import snowflake.connector

def get_snowflake_connection():
    from dotenv import load_dotenv
    load_dotenv()
    print("USER:", os.getenv("SNOWFLAKE_USER"))
    print("PASSWORD:", os.getenv("SNOWFLAKE_PASSWORD"))
    print("ACCOUNT:", os.getenv("SNOWFLAKE_ACCOUNT"))
    print("ROLE:", os.getenv("SNOWFLAKE_ROLE"))
    print("WAREHOUSE:", os.getenv("SNOWFLAKE_WAREHOUSE"))
    print("DATABASE:", os.getenv("SNOWFLAKE_DATABASE"))
    print("SCHEMA:", os.getenv("SNOWFLAKE_SCHEMA"))
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )
    return conn