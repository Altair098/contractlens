from __future__ import annotations

import pandas as pd


def load_dataframe(df: pd.DataFrame, project: str, dataset: str, table: str, location: str = "US") -> int:
    """Append a DataFrame to BigQuery with a batch load job (works in the free BigQuery sandbox)."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    dataset_ref = bigquery.Dataset(f"{project}.{dataset}")
    dataset_ref.location = location
    client.create_dataset(dataset_ref, exists_ok=True)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    job = client.load_table_from_dataframe(df, f"{project}.{dataset}.{table}", job_config=job_config)
    job.result()
    return job.output_rows or 0
