# database_fix.py - Fix the database schema issues

import psycopg2
from psycopg2 import sql
from config import DATABASE_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database_schema():
    """Fix the database schema issues"""
    
    # Parse database URL
    db_url = DATABASE_CONFIG['url']
    # Extract connection details from URL
    # Format: postgresql://username:password@localhost:5432/dbname
    parts = db_url.replace('postgresql://', '').split('/')
    db_name = parts[-1]
    host_part = parts[0]
    
    if '@' in host_part:
        auth_part, host_port = host_part.split('@')
        if ':' in auth_part:
            username, password = auth_part.split(':')
        else:
            username = auth_part
            password = ''
    else:
        username = 'postgres'
        password = ''
        host_port = host_part
    
    if ':' in host_port:
        host, port = host_port.split(':')
    else:
        host = host_port
        port = '5432'
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=db_name,
            user=username,
            password=password
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        logger.info("Connected to database successfully")
        
        # Check if wellness_records table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'wellness_records'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            logger.info("wellness_records table exists, checking schema...")
            
            # Check current columns
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'wellness_records'
                ORDER BY ordinal_position;
            """)
            
            columns = cursor.fetchall()
            logger.info("Current columns in wellness_records:")
            for col in columns:
                logger.info(f"  {col[0]} ({col[1]}) - Nullable: {col[2]}")
            
            # Check if id column exists
            has_id = any(col[0] == 'id' for col in columns)
            
            if not has_id:
                logger.info("Adding missing id column...")
                
                # Drop the table and recreate it properly
                logger.warning("Dropping wellness_records table to recreate with proper schema...")
                cursor.execute("DROP TABLE IF EXISTS wellness_records CASCADE;")
                
                # Create the table with the correct schema
                create_table_sql = """
                CREATE TABLE wellness_records (
                    timestamp TIMESTAMP NOT NULL,
                    astronaut_id VARCHAR(50) NOT NULL,
                    mission_id INTEGER,
                    cognitive_wellness_index FLOAT NOT NULL,
                    stress_level FLOAT NOT NULL,
                    fatigue_level FLOAT NOT NULL,
                    primary_emotion VARCHAR(50),
                    emotion_confidence FLOAT,
                    emotional_stability_score FLOAT,
                    blink_rate FLOAT,
                    eye_closure_duration FLOAT,
                    pupil_dilation FLOAT,
                    vocal_pitch FLOAT,
                    vocal_energy FLOAT,
                    vocal_anomaly_score FLOAT,
                    speech_rate FLOAT,
                    mission_phase VARCHAR(50),
                    activity_type VARCHAR(100),
                    workload_level VARCHAR(20),
                    environmental_stress FLOAT,
                    video_quality_score FLOAT,
                    audio_quality_score FLOAT,
                    sensor_confidence FLOAT,
                    raw_emotion_scores JSON,
                    raw_physiological_data JSON,
                    PRIMARY KEY (timestamp, astronaut_id),
                    FOREIGN KEY (astronaut_id) REFERENCES astronauts(id),
                    FOREIGN KEY (mission_id) REFERENCES missions(id)
                );
                """
                
                cursor.execute(create_table_sql)
                logger.info("wellness_records table recreated with proper schema")
                
                # Create indexes
                index_queries = [
                    "CREATE INDEX IF NOT EXISTS idx_wellness_time_astronaut ON wellness_records(timestamp, astronaut_id);",
                    "CREATE INDEX IF NOT EXISTS idx_wellness_cwi_time ON wellness_records(cognitive_wellness_index, timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_wellness_stress_time ON wellness_records(stress_level, timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_wellness_mission_time ON wellness_records(mission_id, timestamp);"
                ]
                
                for idx_query in index_queries:
                    cursor.execute(idx_query)
                    
                logger.info("Indexes created successfully")
            
            else:
                logger.info("wellness_records table already has correct schema")
        
        else:
            logger.info("wellness_records table doesn't exist, it will be created by SQLAlchemy")
        
        # Verify other tables exist
        tables_to_check = ['astronauts', 'missions', 'wellness_alerts', 'wellness_reports']
        
        for table_name in tables_to_check:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table_name,))
            
            exists = cursor.fetchone()[0]
            logger.info(f"Table {table_name}: {'EXISTS' if exists else 'MISSING'}")
        
        conn.close()
        logger.info("Database schema fix completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing database schema: {e}")
        return False

def test_database_connection():
    """Test database connection and basic operations"""
    try:
        from database_model import EnhancedDatabaseManager, WellnessRecord
        from datetime import datetime
        
        db_manager = EnhancedDatabaseManager(DATABASE_CONFIG['url'])
        
        with db_manager.get_session() as session:
            # Test basic query
            count = session.query(WellnessRecord).count()
            logger.info(f"Current wellness records count: {count}")
            
            # Test inserting a sample record
            test_record = WellnessRecord(
                timestamp=datetime.utcnow(),
                astronaut_id='CREW-001',
                mission_id=1,
                cognitive_wellness_index=75.5,
                stress_level=25.0,
                fatigue_level=20.0,
                primary_emotion='Neutral',
                emotion_confidence=0.8
            )
            
            session.add(test_record)
            session.commit()
            logger.info("Test record inserted successfully!")
            
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print(" V-CARE DATABASE SCHEMA FIX ".center(60))
    print("="*60)
    
    print("\n1. Fixing database schema...")
    if fix_database_schema():
        print("✓ Database schema fixed successfully")
        
        print("\n2. Testing database connection...")
        if test_database_connection():
            print("✓ Database connection test passed")
            print("\n🎉 Database is now ready for V-CARE system!")
        else:
            print("✗ Database connection test failed")
    else:
        print("✗ Database schema fix failed")
    
    print("="*60)