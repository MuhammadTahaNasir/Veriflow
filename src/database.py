import os
import sqlite3
import pandas as pd
import numpy as np

def get_db_connection(db_path="data/inventory.db"):
    """
    Establishes a connection to the SQLite database.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"SQLite database not found at '{db_path}'. "
            "Please ensure you place the downloaded 'inventory.db' file in the 'data/' directory."
        )
    return sqlite3.connect(db_path)

def load_raw_tables(db_path="data/inventory.db"):
    """
    Helper function to load individual tables from the database for exploration.
    """
    conn = get_db_connection(db_path)
    try:
        tables = {}
        for table_name in ["purchases", "vendor_invoice", "purchase_prices"]:
            try:
                tables[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                print(f"Successfully loaded table '{table_name}' with {len(tables[table_name])} records.")
            except Exception as e:
                print(f"Warning: Could not load table '{table_name}': {e}")
        return tables
    finally:
        conn.close()

def load_integrated_audit_data(db_path="data/inventory.db"):
    """
    Loads purchase orders aggregated by PO number and joins them with the
    vendor invoices to create a unified dataset for auditing and ML.
    """
    conn = get_db_connection(db_path)
    try:
        # SQL query:
        # 1. Aggregates the 'purchase' table by PONumber to get total ordered quantity and dollars.
        # 2. Joins it with the 'vendor_invoice' table which represents the billed figures.
        # This matches ordered vs. billed to calculate financial discrepancies.
        query = """
        WITH po_agg AS (
            SELECT 
                PONumber,
                VendorNumber,
                SUM(Quantity) as OrderedQuantity,
                SUM(Dollars) as OrderedDollars,
                COUNT(DISTINCT Brand) as UniqueBrandsOrdered,
                MIN(PODate) as PODate,
                MAX(ReceivingDate) as ReceivingDate
            FROM purchases
            GROUP BY PONumber, VendorNumber
        )
        SELECT 
            vi.PONumber,
            vi.VendorNumber,
            vi.VendorName,
            vi.InvoiceDate,
            vi.PayDate,
            vi.Freight as InvoicedFreight,
            vi.Quantity as InvoicedQuantity,
            vi.Dollars as InvoicedDollars,
            vi.Approval as ApprovalStatus,
            po.OrderedQuantity,
            po.OrderedDollars,
            po.UniqueBrandsOrdered,
            po.PODate,
            po.ReceivingDate
        FROM vendor_invoice vi
        LEFT JOIN po_agg po 
            ON vi.PONumber = po.PONumber 
            AND vi.VendorNumber = po.VendorNumber
        """
        
        df = pd.read_sql_query(query, conn)
        print(f"Successfully loaded and integrated B2B audit dataset: {df.shape[0]} invoices.")
        
        # Fill missing PO values for invoices that have no matching PO in database (if any)
        # using typical B2B values
        df['OrderedQuantity'] = df['OrderedQuantity'].fillna(df['InvoicedQuantity'])
        df['OrderedDollars'] = df['OrderedDollars'].fillna(df['InvoicedDollars'])
        
        return df
        
    except Exception as e:
        print(f"Error executing integrated B2B query: {e}")
        print("Falling back to loading raw vendor_invoice table...")
        # Fallback: Load vendor_invoice directly if purchase table is not present or joined
        fallback_query = "SELECT * FROM vendor_invoice"
        df = pd.read_sql_query(fallback_query, conn)
        
        # Rename columns to match the standard signature
        df = df.rename(columns={
            'Freight': 'InvoicedFreight',
            'Quantity': 'InvoicedQuantity',
            'Dollars': 'InvoicedDollars',
            'Approval': 'ApprovalStatus'
        })
        
        # Synthesize PO columns for backward compatibility if fallback is triggered
        df['OrderedQuantity'] = df['InvoicedQuantity']
        df['OrderedDollars'] = df['InvoicedDollars']
        df['UniqueBrandsOrdered'] = 1
        df['PODate'] = df['InvoiceDate']
        df['ReceivingDate'] = df['InvoiceDate']
        
        return df
    finally:
        conn.close()

if __name__ == "__main__":
    # Test loading
    try:
        data = load_integrated_audit_data()
        print(data.head())
    except Exception as e:
        print(f"Local test check: {e} (This is normal if the database file is not yet uploaded.)")
