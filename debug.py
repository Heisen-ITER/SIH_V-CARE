# debug_database.py - Check if data is actually in your database

import sys
import os
from datetime import datetime, timedelta

# Add your project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from database_model import EnhancedDatabaseManager, WellnessRecord, WellnessAlert, Astronaut, Mission
    from config import DATABASE_CONFIG
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

async def debug_database():
    """Check what's actually in your database."""
    
    try:
        # Connect to database
        db_manager = EnhancedDatabaseManager(DATABASE_CONFIG['url'])
        print("✓ Database connection established")
        
        with db_manager.get_session() as session:
            # Check astronauts
            astronauts = session.query(Astronaut).all()
            print(f"✓ Found {len(astronauts)} astronauts:")
            for ast in astronauts:
                print(f"  - {ast.id}: {ast.full_name}")
            
            # Check missions
            missions = session.query(Mission).all()
            print(f"✓ Found {len(missions)} missions:")
            for mission in missions:
                print(f"  - {mission.mission_code}: {mission.mission_type}")
            
            # Check wellness records
            wellness_count = session.query(WellnessRecord).count()
            print(f"✓ Total wellness records: {wellness_count}")
            
            if wellness_count > 0:
                # Show latest 5 records
                latest_records = session.query(WellnessRecord).order_by(
                    WellnessRecord.timestamp.desc()
                ).limit(5).all()
                
                print("\n📊 Latest 5 wellness records:")
                for i, record in enumerate(latest_records, 1):
                    print(f"  {i}. {record.timestamp} - CWI: {record.cognitive_wellness_index}%, "
                          f"Stress: {record.stress_level}%, Fatigue: {record.fatigue_level}%")
                
                # Show oldest record
                oldest = session.query(WellnessRecord).order_by(
                    WellnessRecord.timestamp.asc()
                ).first()
                print(f"\n🕐 Oldest record: {oldest.timestamp}")
                
                # Show newest record
                newest = session.query(WellnessRecord).order_by(
                    WellnessRecord.timestamp.desc()
                ).first()
                print(f"🕐 Newest record: {newest.timestamp}")
                
            else:
                print("❌ No wellness records found in database!")
                print("   This explains why your PDF is empty.")
            
            # Check alerts
            alerts_count = session.query(WellnessAlert).count()
            print(f"✓ Total alerts: {alerts_count}")
            
            if alerts_count > 0:
                latest_alerts = session.query(WellnessAlert).order_by(
                    WellnessAlert.timestamp.desc()
                ).limit(3).all()
                print("\n🚨 Latest 3 alerts:")
                for alert in latest_alerts:
                    print(f"  - {alert.timestamp}: {alert.alert_type} ({alert.severity})")
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    print("🔍 Debugging V-CARE Database...")
    print("="*50)
    asyncio.run(debug_database())
    print("="*50)
    print("Debug complete!")