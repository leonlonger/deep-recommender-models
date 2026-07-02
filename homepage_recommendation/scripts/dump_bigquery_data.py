"""Dump homepage recommendation training data from BigQuery to local Parquet."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.auth import _cloud_sdk
from google.auth import credentials as google_credentials
from google.auth.exceptions import RefreshError
from google.cloud import bigquery

try:
    import yaml
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit("Missing dependency PyYAML. Install dependencies with: pip install -r requirements.txt") from exc


DEFAULT_OUTPUT_PATH = Path("/mnt/disk/datasets/homepage_training_examples.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to training config YAML.")
    parser.add_argument(
        "--output",
        help=(
            "Output Parquet path. Defaults to data.path in config.yaml, "
            "then /mnt/disk/datasets/homepage_training_examples.parquet."
        ),
    )
    parser.add_argument("--limit", type=int, help="Optional row limit for a small local dump.")
    parser.add_argument("--page-size", type=int, default=50_000, help="Rows per fallback BigQuery result page.")
    parser.add_argument("--progress-rows", type=int, default=100_000, help="Rows between progress messages.")
    parser.add_argument(
        "--row-group-rows",
        type=int,
        default=250_000,
        help="Target rows per Parquet row group when compacting BigQuery batches.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec. Defaults to zstd.",
    )
    parser.add_argument(
        "--no-compact-row-groups",
        action="store_true",
        help="Write each BigQuery Arrow batch directly, matching the old row group behavior.",
    )
    parser.add_argument(
        "--max-stream-count",
        type=int,
        default=1,
        help="Maximum BigQuery Storage API streams. Keep this low on small VMs.",
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=1,
        help="Maximum queued BigQuery Storage API pages. Keep this low on small VMs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate the query without writing data.")
    parser.add_argument(
        "--auth-mode",
        choices=("auto", "adc", "gcloud"),
        default="auto",
        help="Authentication mode. auto tries ADC first, then falls back to the active gcloud account.",
    )
    parser.add_argument("--gcloud-account", help="Optional gcloud account email for --auth-mode=gcloud.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]


def quote_bigquery_identifier(identifier: str) -> str:
    if "`" in identifier:
        raise ValueError(f"Backticks are not allowed in BigQuery identifiers: {identifier}")
    return f"`{identifier}`"


def get_bigquery_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    bigquery_config = data_config.get("bigquery")
    if isinstance(bigquery_config, dict):
        return bigquery_config
    return data_config


def build_bigquery_sql(bigquery_config: dict[str, Any], *, row_limit: int | None) -> str:
    custom_query = bigquery_config.get("query")
    if custom_query:
        sql = str(custom_query)
    else:
        table_id = bigquery_config.get("table_id")
        if not table_id:
            raise ValueError("data.bigquery.table_id is required.")

        selected_columns = as_string_list(bigquery_config.get("selected_columns"))
        if selected_columns:
            select_clause = ", ".join(quote_bigquery_identifier(column) for column in selected_columns)
        else:
            select_clause = "*"

        sql = f"SELECT {select_clause}\nFROM {quote_bigquery_identifier(str(table_id))}"
        where_clause = bigquery_config.get("where") or bigquery_config.get("where_clause")
        if where_clause:
            sql += f"\nWHERE {where_clause}"

    if row_limit:
        sql += f"\nLIMIT {int(row_limit)}"
    return sql


def can_read_table_directly(bigquery_config: dict[str, Any], *, row_limit: int | None) -> bool:
    return not (
        bigquery_config.get("query")
        or bigquery_config.get("where")
        or bigquery_config.get("where_clause")
        or row_limit
    )


def resolve_output_path(config: dict[str, Any], args: argparse.Namespace, *, config_path: Path) -> Path:
    output_value = args.output or config.get("data", {}).get("path") or DEFAULT_OUTPUT_PATH
    output_path = Path(str(output_value)).expanduser()
    if not output_path.is_absolute():
        output_path = config_path.parent / output_path
    return output_path


def selected_table_fields(
    table: bigquery.Table,
    selected_columns: list[str],
) -> list[bigquery.SchemaField] | None:
    if not selected_columns:
        return None

    fields_by_name = {field.name: field for field in table.schema}
    missing = [column for column in selected_columns if column not in fields_by_name]
    if missing:
        raise ValueError(f"Selected BigQuery columns do not exist in table schema: {missing}")
    return [fields_by_name[column] for column in selected_columns]


class CloudSdkCredentials(google_credentials.Credentials):
    """Refreshable credentials backed by `gcloud auth print-access-token`."""

    def __init__(self, account: str | None = None) -> None:
        super().__init__()
        self.account = account

    def refresh(self, request: Any) -> None:
        self.token = _cloud_sdk.get_auth_access_token(account=self.account)
        self.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=50)


def build_bigquery_client(
    *,
    project_id: str | None,
    auth_mode: str,
    gcloud_account: str | None,
) -> bigquery.Client:
    if auth_mode == "gcloud":
        credentials = CloudSdkCredentials(account=gcloud_account)
        credentials.refresh(None)
        return bigquery.Client(project=project_id, credentials=credentials)
    return bigquery.Client(project=project_id)


def build_bqstorage_client(
    credentials: google_credentials.Credentials | None,
) -> Any:
    try:
        from google.cloud import bigquery_storage
    except ImportError:
        return None
    return bigquery_storage.BigQueryReadClient(credentials=credentials)


def iter_arrow_tables(
    row_iterator: bigquery.table.RowIterator,
    *,
    page_size: int,
    credentials: google_credentials.Credentials | None,
    max_stream_count: int,
    max_queue_size: int,
) -> Any:
    bqstorage_client = build_bqstorage_client(credentials)

    if hasattr(row_iterator, "to_arrow_iterable"):
        batches = row_iterator.to_arrow_iterable(
            bqstorage_client=bqstorage_client,
            max_queue_size=max_queue_size,
            max_stream_count=max_stream_count,
        )
        for batch in batches:
            if batch.num_rows:
                yield pa.Table.from_batches([batch])
        return

    for page in row_iterator.pages:
        rows = [dict(row.items()) for row in page]
        if rows:
            dataframe = pd.DataFrame(rows)
            yield pa.Table.from_pandas(dataframe, preserve_index=False)


def write_parquet_chunks(
    chunks: Any,
    output_path: Path,
    *,
    progress_rows: int,
    row_group_rows: int,
    compression: str,
    compact_row_groups: bool,
) -> int:
    if compact_row_groups and row_group_rows <= 0:
        raise ValueError("--row-group-rows must be positive unless --no-compact-row-groups is set.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    next_progress = max(progress_rows, 1)
    buffered_tables: list[pa.Table] = []
    buffered_rows = 0

    def ensure_writer(schema: pa.Schema) -> pq.ParquetWriter:
        nonlocal writer
        if writer is None:
            writer = pq.ParquetWriter(
                temporary_path,
                schema,
                compression=compression,
            )
        return writer

    def record_written_rows(row_count: int) -> None:
        nonlocal total_rows, next_progress
        total_rows += row_count
        if total_rows >= next_progress:
            print(f"Wrote {total_rows} rows...")
            while total_rows >= next_progress:
                next_progress += max(progress_rows, 1)

    def flush_buffer(*, final: bool = False) -> None:
        nonlocal buffered_tables, buffered_rows
        if not buffered_rows:
            return

        combined = pa.concat_tables(buffered_tables)
        rows_to_write = combined.num_rows if final else (combined.num_rows // row_group_rows) * row_group_rows
        if rows_to_write <= 0:
            return

        table_to_write = combined.slice(0, rows_to_write)
        ensure_writer(table_to_write.schema).write_table(
            table_to_write,
            row_group_size=row_group_rows,
        )
        record_written_rows(rows_to_write)

        remaining_rows = combined.num_rows - rows_to_write
        if remaining_rows:
            buffered_tables = [combined.slice(rows_to_write, remaining_rows)]
            buffered_rows = remaining_rows
        else:
            buffered_tables = []
            buffered_rows = 0

    try:
        for table in chunks:
            if table.num_rows == 0:
                continue
            if not compact_row_groups:
                ensure_writer(table.schema).write_table(table)
                record_written_rows(table.num_rows)
                continue

            buffered_tables.append(table)
            buffered_rows += table.num_rows
            flush_buffer()
        if compact_row_groups:
            flush_buffer(final=True)
    finally:
        if writer is not None:
            writer.close()
    if total_rows:
        temporary_path.replace(output_path)
    return total_rows


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    bigquery_config = get_bigquery_config(config)
    project_id = bigquery_config.get("project_id")
    output_path = resolve_output_path(config, args, config_path=config_path)
    sql = build_bigquery_sql(bigquery_config, row_limit=args.limit)
    compact_row_groups = not args.no_compact_row_groups

    client = build_bigquery_client(
        project_id=project_id,
        auth_mode="adc" if args.auth_mode == "auto" else args.auth_mode,
        gcloud_account=args.gcloud_account,
    )

    def run_dump(active_client: bigquery.Client) -> int | None:
        if args.dry_run:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            dry_run_job = active_client.query(sql, job_config=job_config)
            print("Dry run query:")
            print(sql)
            print(f"Estimated bytes processed: {dry_run_job.total_bytes_processed}")
            return None

        if can_read_table_directly(bigquery_config, row_limit=args.limit):
            table_id = bigquery_config.get("table_id")
            if not table_id:
                raise ValueError("data.bigquery.table_id is required.")
            table = active_client.get_table(str(table_id))
            selected_fields = selected_table_fields(
                table,
                as_string_list(bigquery_config.get("selected_columns")),
            )
            print("Reading BigQuery table directly:")
            print(f"  table: {table_id}")
            print(f"  selected_columns: {len(selected_fields or table.schema)}")
            print(f"Writing local Parquet: {output_path}")
            print(f"  compression: {args.compression}")
            if compact_row_groups:
                print(f"  target row group rows: {args.row_group_rows}")
            else:
                print("  row group compaction: disabled")
            row_iterator = active_client.list_rows(
                table,
                selected_fields=selected_fields,
                page_size=args.page_size,
            )
        else:
            print("Running BigQuery dump query:")
            print(sql)
            print(f"Writing local Parquet: {output_path}")
            print(f"  compression: {args.compression}")
            if compact_row_groups:
                print(f"  target row group rows: {args.row_group_rows}")
            else:
                print("  row group compaction: disabled")
            query_job = active_client.query(sql)
            row_iterator = query_job.result(page_size=args.page_size)
        chunks = iter_arrow_tables(
            row_iterator,
            page_size=args.page_size,
            credentials=active_client._credentials,
            max_stream_count=args.max_stream_count,
            max_queue_size=args.max_queue_size,
        )
        return write_parquet_chunks(
            chunks,
            output_path,
            progress_rows=args.progress_rows,
            row_group_rows=args.row_group_rows,
            compression=args.compression,
            compact_row_groups=compact_row_groups,
        )

    try:
        total_rows = run_dump(client)
    except RefreshError as exc:
        if args.auth_mode != "auto":
            raise SystemExit(
                "BigQuery authentication needs to be refreshed. Run:\n"
                "  gcloud auth application-default login "
                "--scopes=https://www.googleapis.com/auth/cloud-platform,"
                "https://www.googleapis.com/auth/bigquery,"
                "https://www.googleapis.com/auth/userinfo.email,openid"
            ) from exc
        print("ADC authentication needs refresh; falling back to active gcloud account.")
        fallback_client = build_bigquery_client(
            project_id=project_id,
            auth_mode="gcloud",
            gcloud_account=args.gcloud_account,
        )
        total_rows = run_dump(fallback_client)
    if total_rows is None:
        return
    print(f"Done. Wrote {total_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
