# banking_api.py
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- DB Setup -----
def init_db():
    conn = sqlite3.connect("banking.db")
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS accounts (
        accountno TEXT PRIMARY KEY,
        balance REAL
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        accountno TEXT,
        description TEXT,
        amount REAL,
        type TEXT
    )''')

    # Insert dummy data
    accounts = [
        ("12345", 34500.75),
        ("67890", 15200.10),
        ("11111", 9025.00),
    ]
    cursor.executemany("INSERT OR IGNORE INTO accounts VALUES (?, ?)", accounts)

    transactions = [
        ("12345", "POS Purchase", -1200, "debit"),
        ("12345", "ATM Withdrawal", -850, "debit"),
        ("12345", "Salary Credit", 5000, "credit"),
        ("67890", "Online Transfer", -2000, "debit"),
        ("11111", "UPI Payment", -500, "debit")
    ]
    cursor.executemany("INSERT INTO transactions (accountno, description, amount, type) VALUES (?, ?, ?, ?)", transactions)

    conn.commit()
    conn.close()

init_db()

# ----- Endpoints -----
@app.get("/balance")
def get_balance(accountno: str):
    conn = sqlite3.connect("banking.db")
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM accounts WHERE accountno = ?", (accountno,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return {"accountno": accountno, "balance": result[0]}
    else:
        raise HTTPException(status_code=404, detail="Account not found")

@app.get("/transactions")
def get_transactions(accountno: str):
    conn = sqlite3.connect("banking.db")
    cursor = conn.cursor()
    cursor.execute("SELECT description, amount, type FROM transactions WHERE accountno = ? ORDER BY id DESC LIMIT 5", (accountno,))
    results = cursor.fetchall()
    conn.close()

    if results:
        return {
            "accountno": accountno,
            "transactions": [{"desc": r[0], "amount": r[1], "type": r[2]} for r in results]
        }
    else:
        raise HTTPException(status_code=404, detail="No transactions found")

# Run locally with: python banking_api.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
