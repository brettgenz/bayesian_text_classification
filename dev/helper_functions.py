import pandas as pd

def glimpse(df, width=80):
    """
    Mimic dplyr::glimpse() for a pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to display.
        width (int): Maximum width for row previews.
    """
    n_rows, n_cols = df.shape
    print(f"Rows: {n_rows}, Columns: {n_cols}\n")
    
    for col in df.columns:
        dtype = df[col].dtype
        preview = df[col].astype(str).head(5).tolist()  # show first 5 values
        preview_str = ", ".join(preview)
        
        if len(preview_str) > width:
            preview_str = preview_str[: width - 3] + "..."
        
        print(f"${col} <{dtype}> {preview_str}")
