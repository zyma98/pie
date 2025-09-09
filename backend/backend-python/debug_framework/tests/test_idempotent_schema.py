#!/usr/bin/env python3
"""
Test to verify that the schema can be run multiple times without errors (idempotent).
"""
import sqlite3
import tempfile
import os

def test_idempotent_schema():
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Read schema file
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'schema.sql')
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Create database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("Test 1: Running schema for the first time...")
        cursor.executescript(schema_sql)
        print("✓ Schema applied successfully on first run")

        print("\nTest 2: Running schema for the second time...")
        cursor.executescript(schema_sql)
        print("✓ Schema applied successfully on second run (idempotent)")

        print("\nTest 3: Running schema for the third time...")
        cursor.executescript(schema_sql)
        print("✓ Schema applied successfully on third run (idempotent)")

        # Verify that tables exist and work correctly after multiple runs
        print("\nTest 4: Verifying functionality after multiple schema runs...")
        cursor.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model",))
        cursor.execute("SELECT COUNT(*) FROM debug_sessions")
        count = cursor.fetchone()[0]
        print(f"✓ Tables functional after multiple runs (session count: {count})")

        conn.commit()
        conn.close()

        print("\n✅ Idempotency test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Idempotency test failed: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    test_idempotent_schema()