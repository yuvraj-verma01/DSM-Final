import os
import re
import subprocess
import sys
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def _in_streamlit_runtime() -> bool:
    """Return True when this script is executing inside Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _in_streamlit_runtime():
    print("Launching Streamlit app...", flush=True)
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
    raise SystemExit(0)


load_dotenv()

DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL",
    "mysql+mysqlconnector://root:@localhost:3306/dsm_final_project",
)
DB_SCHEMA = os.getenv("DB_SCHEMA", "dsm_final_project")
DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "tencent/hy3-preview:free")
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in your .env file before running the agent.")

    return OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)


@st.cache_resource(show_spinner=False)
def get_engine(db_url: str) -> Engine:
    return create_engine(db_url)


def test_database_connection(db_url: str) -> None:
    engine = get_engine(db_url)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def list_tables(engine: Engine, schema_name: str = DB_SCHEMA) -> pd.DataFrame:
    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema_name
        ORDER BY table_name
        """
    )
    return pd.read_sql(query, engine, params={"schema_name": schema_name})


def describe_table(engine: Engine, table_name: str, schema_name: str = DB_SCHEMA) -> pd.DataFrame:
    query = text(
        """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_key
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        ORDER BY ordinal_position
        """
    )
    return pd.read_sql(
        query,
        engine,
        params={"schema_name": schema_name, "table_name": table_name},
    )


@st.cache_data(show_spinner=False)
def build_schema_context(db_url: str, schema_name: str = DB_SCHEMA) -> str:
    engine = get_engine(db_url)
    table_names = list_tables(engine, schema_name)["table_name"].tolist()
    sections = []

    for table_name in table_names:
        columns_df = describe_table(engine, table_name, schema_name)
        column_lines = [
            f"- {row.column_name} ({row.data_type}, nullable={row.is_nullable}, key={row.column_key or 'none'})"
            for row in columns_df.itertuples(index=False)
        ]
        sections.append(f"Table: {table_name}\n" + "\n".join(column_lines))

    return "\n\n".join(sections)


def clean_sql(sql_text: str) -> str:
    sql_text = sql_text.strip()
    sql_text = re.sub(r"^```sql\s*|^```\s*|\s*```$", "", sql_text, flags=re.IGNORECASE)
    return sql_text.strip()


def is_safe_select_query(query: str) -> bool:
    cleaned = clean_sql(query)
    lowered = cleaned.lower()
    lowered = re.sub(r"--.*?$", "", lowered, flags=re.MULTILINE)
    lowered = re.sub(r"/\*.*?\*/", "", lowered, flags=re.DOTALL)
    stripped = lowered.strip().rstrip(";")

    if not stripped.startswith(("select", "with")):
        return False

    blocked_terms = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "truncate ",
        "create ",
        "replace ",
        "grant ",
        "revoke ",
    ]
    return not any(term in stripped for term in blocked_terms)


def safe_run_sql(engine: Engine, query: str, row_limit: int = 200) -> pd.DataFrame:
    cleaned = clean_sql(query).rstrip(";")
    if not is_safe_select_query(cleaned):
        raise ValueError("Only read-only SELECT queries are allowed.")

    limited_query = f"SELECT * FROM ({cleaned}) AS analyst_query LIMIT {row_limit}"
    return pd.read_sql(text(limited_query), engine)


def generate_sql(question: str, schema_context: str, model: str = DEFAULT_MODEL) -> str:
    client = get_openai_client()
    prompt = f"""
You are a careful MySQL data analyst.

Database schema:
{schema_context}

User question:
{question}

Rules:
- Return only SQL.
- Use only tables and columns from the schema.
- Produce a single read-only query.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
- Prefer explicit column names instead of SELECT * unless it is necessary.
- If aggregation is needed, include clear aliases.
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "Return only valid MySQL SQL."}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
    )
    return clean_sql(response.output_text)


def summarize_results(
    question: str,
    sql_query: str,
    result_df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
) -> str:
    client = get_openai_client()
    preview_csv = result_df.head(20).to_csv(index=False)

    prompt = f"""
You are summarizing the output of a SQL analysis.

User question:
{question}

SQL used:
{sql_query}

Preview of result rows:
{preview_csv}

Write a concise answer grounded only in the result preview.
If the preview is insufficient, say that more rows may be needed.
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text.strip()


def ask_database(
    question: str,
    db_url: str,
    schema_name: str = DB_SCHEMA,
    model: str = DEFAULT_MODEL,
    row_limit: int = 200,
    return_summary: bool = True,
) -> dict[str, Any]:
    engine = get_engine(db_url)
    schema_context = build_schema_context(db_url, schema_name)
    sql_query = generate_sql(question, schema_context, model=model)
    result_df = safe_run_sql(engine, sql_query, row_limit=row_limit)

    response: dict[str, Any] = {
        "question": question,
        "sql_query": sql_query,
        "rows_returned": len(result_df),
        "data": result_df,
    }

    if return_summary:
        response["summary"] = summarize_results(question, sql_query, result_df, model=model)

    return response


st.set_page_config(page_title="DSM SQL Analyst", page_icon=":bar_chart:", layout="wide")
st.title("DSM SQL Analyst")
st.caption("Natural-language questions over your MySQL database.")

with st.sidebar:
    st.header("Database")
    db_url = st.text_input("Database URL", value=DEFAULT_DB_URL, type="password")
    schema_name = st.text_input("Schema", value=DB_SCHEMA)

    st.header("Model")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    row_limit = st.slider("Row limit", min_value=10, max_value=1000, value=200, step=10)
    return_summary = st.checkbox("Generate summary", value=True)

    if st.button("Test database connection"):
        try:
            test_database_connection(db_url)
            st.success("Database connection successful.")
        except Exception as exc:
            st.error(f"Database connection failed: {exc}")

    if st.button("Preview tables"):
        try:
            engine = get_engine(db_url)
            tables_df = list_tables(engine, schema_name)
            if tables_df.empty:
                st.warning(f"No tables found in schema `{schema_name}`.")
            else:
                st.dataframe(tables_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load tables: {exc}")


question = st.text_area(
    "Ask a question about your database",
    placeholder="Which states have the highest literacy_total_pct?",
    height=120,
)

if st.button("Run analysis", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Inspecting schema, generating SQL, and querying MySQL..."):
                result = ask_database(
                    question=question.strip(),
                    db_url=db_url,
                    schema_name=schema_name,
                    model=model,
                    row_limit=row_limit,
                    return_summary=return_summary,
                )

            if return_summary and result.get("summary"):
                st.subheader("Summary")
                st.write(result["summary"])

            st.subheader("Generated SQL")
            st.code(result["sql_query"], language="sql")

            st.subheader(f"Results ({result['rows_returned']} rows)")
            st.dataframe(result["data"], use_container_width=True)

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
