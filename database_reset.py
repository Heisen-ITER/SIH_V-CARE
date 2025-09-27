# simple_database_reset.py - Reset database with correct schema

import asyncio
from database_model import EnhancedDatabaseManager, Base
from config import DATABASE_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reset_database():
    """Reset database with correct schema"""
    
    try:
        # Create database manager
        db_manager = EnhancedDatabaseManager(
            DATABASE_CONFIG['url'], 
            enable_timescale=False  # Disable TimescaleDB for now since it's not installed
        )
        
        logger.info("Dropping all existing tables...")
        
        # Drop all tables
        Base.metadata.drop_all(bind=db_manager.engine)
        logger.info("All tables dropped successfully")
        
        # Create all tables with correct schema
        logger.info("Creating tables with correct schema...")
        Base.metadata.create_all(bind=db_manager.engine)
        logger.info("All tables created successfully")
        
        # Initialize basic data
        from database_model import Astronaut, Mission
        from datetime import datetime
        
        with db_manager.get_session() as session:
            # Create astronaut
            astronaut = Astronaut(
                id="CREW-001",
                full_name="V-CARE Test Subject",
                baseline_stress_threshold=70.0,
                baseline_fatigue_threshold=65.0,
                baseline_cwi_score=85.0,
                psychological_profile={
                    "stress_resilience": "high",
                    "fatigue_tolerance": "moderate",
                    "emotional_stability": "excellent"
                }
            )
            session.add(astronaut)
            
            # Create mission
            mission = Mission(
                mission_code=f"V-CARE-DEMO-{datetime.now().strftime('%Y%m%d')}",
                astronaut_id="CREW-001",
                mission_type="Cognitive Wellness Monitoring",
                start_date=datetime.utcnow(),
                current_phase="Active Monitoring",
                duration_hours=24,
                mission_objectives="Real-time cognitive wellness monitoring demonstration",
                risk_level="low"
            )
            session.add(mission)
            
            session.commit()
            logger.info("Basic astronaut and mission data created")
        
        # Test the schema
        with db_manager.get_session() as session:
            from database_model import WellnessRecord
            
            # Try to query - this will fail if schema is wrong
            count = session.query(WellnessRecord).count()
            logger.info(f"Wellness records table is working. Current count: {count}")
            
            # Insert a test record to verify everything works
            test_record = WellnessRecord(
                timestamp=datetime.utcnow(),
                astronaut_id='CREW-001',
                mission_id=1,
                cognitive_wellness_index=75.5,
                stress_level=25.0,
                fatigue_level=20.0,
                primary_emotion='Neutral',
                emotion_confidence=0.8,
                blink_rate=15.0,
                vocal_pitch=180.0,
                vocal_energy=0.001,
                vocal_anomaly_score=0.2
            )
            
            session.add(test_record)
            session.commit()
            
            # Verify the record was saved
            count_after = session.query(WellnessRecord).count()
            logger.info(f"Test record inserted. New count: {count_after}")
        
        logger.info("Database reset completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        logger.error(f"Full error: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print(" V-CARE DATABASE RESET ".center(60))
    print("="*60)
    print("\nThis will:")
    print("1. Drop all existing tables")
    print("2. Create new tables with correct schema")
    print("3. Insert basic astronaut and mission data")
    print("4. Test the database functionality")
    
    response = input("\nContinue? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        result = asyncio.run(reset_database())
        
        if result:
            print("\n✅ DATABASE RESET SUCCESSFUL!")
            print("\nYour database is now ready for the V-CARE system.")
            print("You can now run main.py and generate reports successfully.")
        else:
            print("\n❌ DATABASE RESET FAILED!")
            print("Please check the error messages above.")
    else:
        print("Database reset cancelled.")
    
    print("="*60)