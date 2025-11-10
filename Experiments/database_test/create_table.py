import psycopg2

conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)

cur = conn.cursor()

# cur.execute("""CREATE TABLE imu_data(
#             id SERIAL PRIMARY KEY,
#             speed DECIMAL(10, 3) NOT NULL,
#             yaw_rate_dps DECIMAL(10, 3) NOT NULL,
#             turn_angle_deg DECIMAL(10, 3) NOT NULL,
#             left_speed DECIMAL(10, 3) NOT NULL,
#             right_speed DECIMAL(10, 3) NOT NULL,
#             timestamp TIMESTAMPTZ NOT NULL
#             );
#             """)

cur.execute("""
            CREATE TABLE test_table(
            id SERIAL PRIMARY KEY,
            distance DECIMAL(10, 3) NOT NULL,
            command varchar(255)
            );
            """)

conn.commit()

cur.close()
conn.close()
