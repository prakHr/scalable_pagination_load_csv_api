import os

os.environ["OMP_NUM_THREADS"] = "1"

import math
import multiprocessing
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from starlette.middleware.gzip import GZipMiddleware
from mpire import WorkerPool


filename = os.path.basename(__file__).split(".")[0]

app = FastAPI(
    title=filename,
    version="1.0.0"
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------
# CSV READER
# ---------------------------------------------------------

def read_csv_safely(csv_path: str):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    for encoding in encodings:

        try:
            return pd.read_csv(
                csv_path,
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    return pd.read_csv(
        csv_path,
        encoding="latin1",
        encoding_errors="replace"
    )


# ---------------------------------------------------------
# WORKER
# ---------------------------------------------------------

def get_dataframe(csv_path,page_from,page_to,page_size):
    try:

        df = read_csv_safely(csv_path)

        total_rows = len(df)

        total_pages = math.ceil(
            total_rows / page_size
        )

        # Don't allow page_to beyond available pages
        actual_page_to = min(
            page_to,
            total_pages
        )

        pages = []

        for page in range(
            page_from,
            actual_page_to + 1
        ):

            offset = (page - 1) * page_size

            paginated_df = df.iloc[
                offset:offset + page_size
            ]

            # NaN / NaT -> None
            paginated_df = (
                paginated_df
                .astype(object)
                .where(
                    pd.notna(paginated_df),
                    None
                )
            )

            pages.append({
                "page": page,
                "offset": offset,
                "rows": len(paginated_df),
                "data": paginated_df.to_dict(
                    orient="records"
                )
            })

        return {
            "success": True,
            "csv_path": csv_path,
            "page_from": page_from,
            "page_to": actual_page_to,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "columns": df.columns.tolist(),
            "pages": pages
        }

    except Exception as e:

        return {
            "success": False,
            "csv_path": csv_path,
            "error": str(e)
        }


# ---------------------------------------------------------
# MULTIPLE CSV FILES
# ---------------------------------------------------------

def remove_duplicate_csv_paths(csv_paths):
    csv_paths2 = []
    st = set()
    for csv_path in csv_paths:
        if csv_path not in st:
            csv_paths2.append(csv_path)
            st.add(csv_path)
    csv_paths = csv_paths2
    return csv_paths

@app.get("/get_data_from_csv_files")
def get_data_from_csv_files(

    page_from: int = Query(
        1,
        ge=1,
        description="Starting page number"
    ),

    page_to: int = Query(
        1,
        ge=1,
        description="Ending page number"
    ),

    progress_bar: bool = False,

    csv_paths: List[str] = Query(...)
):

    page_size = 1
    # Validate page range
    if page_to < page_from:

        raise HTTPException(
            status_code=400,
            detail="page_to must be greater than or equal to page_from"
        )

    if abs(page_from - page_to)>10:
        raise HTTPException(
            status_code=400,
            detail="keep the range of page_from and page_to between 1 - 10"
        )
        

    num_cores = max(
        min(
            multiprocessing.cpu_count() // 2,
            2
        ),
        1
    )

    tasks = []
    csv_paths = remove_duplicate_csv_paths(csv_paths)
    
    for csv_path in csv_paths:

        tasks.append({
            "csv_path": csv_path,
            "page_from": page_from,
            "page_to": page_to,
            "page_size": page_size
        })

    with WorkerPool(
        n_jobs=num_cores,
        daemon=False
    ) as pool:

        results = pool.map(
            get_dataframe,
            tasks,
            progress_bar=progress_bar
        )

    return {
        "page_from": page_from,
        "page_to": page_to,
        "page_size": page_size,
        "files": results
    }