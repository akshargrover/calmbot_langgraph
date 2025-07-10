import sqlite3

def seed_data():
    conn = sqlite3.connect("data/therapist.db")
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS therapists;
    DROP TABLE IF EXISTS availability;
    DROP TABLE IF EXISTS user_logs;

    CREATE TABLE therapists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        specialty TEXT NOT NULL
    );

    CREATE TABLE availability (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        therapist_id INTEGER NOT NULL,
        slot TEXT NOT NULL,
        FOREIGN KEY (therapist_id) REFERENCES therapists(id)
    );

    CREATE TABLE user_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        emotion TEXT,
        therapist_name TEXT,
        slot TEXT,
        booked_at TEXT
    );
                      
    CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        therapist_id INTEGER,
        therapist_name TEXT,
        slot TEXT,
        user_id TEXT,
        emotion TEXT,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Insert therapists
    therapists = [
        ("Dr. Meera Kapoor", "anxiety,depression"),
        ("Dr. Aman Verma", "grief,stress"),
        ("Dr. Kavita Shah", "joy,gratitude,confidence"),
    ]
    cur.executemany("INSERT INTO therapists (name, specialty) VALUES (?, ?)", therapists)

    # Insert availability
    availability = [
        (1, "2025-06-28 10:00"),
        (1, "2025-06-28 15:00"),
        (2, "2025-06-27 18:00"),
        (2, "2025-06-28 12:30"),
        (3, "2025-06-29 09:00"),
    ]
    cur.executemany("INSERT INTO availability (therapist_id, slot) VALUES (?, ?)", availability)

    conn.commit()
    conn.close()

    

if __name__ == "__main__":
    seed_data()
