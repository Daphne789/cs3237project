import psycopg2

conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)

cur = conn.cursor()

cur.execute("""CREATE TABLE movement_data(
            id SERIAL PRIMARY KEY,
            direction VARCHAR (50) UNIQUE NOT NULL,
            speed DECIMAL(10, 2) UNIQUE NOT NULL
            );
            """)

conn.commit()

cur.close()
conn.close()
