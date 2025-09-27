# enhanced_database_models.py - PostgreSQL + TimescaleDB optimized (FIXED VERSION)

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func, text
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

class Astronaut(Base):
    __tablename__ = 'astronauts'
    
    id = Column(String(50), primary_key=True)
    full_name = Column(String(200), nullable=False)
    date_of_birth = Column(DateTime, nullable=True)
    medical_clearance_date = Column(DateTime, nullable=True)
    baseline_stress_threshold = Column(Float, default=70.0)
    baseline_fatigue_threshold = Column(Float, default=65.0)
    baseline_cwi_score = Column(Float, default=85.0)
    psychological_profile = Column(JSON, nullable=True)
    medical_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    missions = relationship("Mission", back_populates="astronaut")
    wellness_records = relationship("WellnessRecord", back_populates="astronaut")
    alerts = relationship("WellnessAlert", back_populates="astronaut")
    reports = relationship("WellnessReport", back_populates="astronaut")

class Mission(Base):
    __tablename__ = 'missions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_code = Column(String(100), unique=True, nullable=False)
    astronaut_id = Column(String(50), ForeignKey('astronauts.id'), nullable=False)
    mission_type = Column(String(50), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    current_phase = Column(String(50), nullable=True)
    duration_hours = Column(Float, nullable=True)
    mission_objectives = Column(Text, nullable=True)
    risk_level = Column(String(20), default='moderate')
    environmental_factors = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    astronaut = relationship("Astronaut", back_populates="missions")
    wellness_records = relationship("WellnessRecord", back_populates="mission")
    alerts = relationship("WellnessAlert", back_populates="mission")
    reports = relationship("WellnessReport", back_populates="mission")

# FIXED: Now uses single primary key instead of composite
class WellnessRecord(Base):
    __tablename__ = 'wellness_records'
    
    # FIXED: Single primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=func.now(), index=True)
    astronaut_id = Column(String(50), ForeignKey('astronauts.id'), nullable=False, index=True)
    
    # Secondary fields
    mission_id = Column(Integer, ForeignKey('missions.id'), nullable=True)
    
    # Core wellness metrics - optimized for analytics
    cognitive_wellness_index = Column(Float, nullable=False, index=True)
    stress_level = Column(Float, nullable=False, index=True)
    fatigue_level = Column(Float, nullable=False, index=True)
    
    # Emotional analysis
    primary_emotion = Column(String(50), nullable=True, index=True)
    emotion_confidence = Column(Float, nullable=True)
    emotional_stability_score = Column(Float, nullable=True)
    
    # Physiological metrics
    blink_rate = Column(Float, nullable=True)
    eye_closure_duration = Column(Float, nullable=True)
    pupil_dilation = Column(Float, nullable=True)
    
    # Vocal analysis metrics
    vocal_pitch = Column(Float, nullable=True)
    vocal_energy = Column(Float, nullable=True)
    vocal_anomaly_score = Column(Float, nullable=True)
    speech_rate = Column(Float, nullable=True)
    
    # Contextual information
    mission_phase = Column(String(50), nullable=True)
    activity_type = Column(String(100), nullable=True)
    workload_level = Column(String(20), nullable=True)
    environmental_stress = Column(Float, nullable=True)
    
    # Data quality metrics
    video_quality_score = Column(Float, nullable=True)
    audio_quality_score = Column(Float, nullable=True)
    sensor_confidence = Column(Float, nullable=True)
    
    # Raw analysis results stored as JSONB for efficient querying
    raw_emotion_scores = Column(JSON, nullable=True)
    raw_physiological_data = Column(JSON, nullable=True)
    
    # Relationships
    astronaut = relationship("Astronaut", back_populates="wellness_records")
    mission = relationship("Mission", back_populates="wellness_records")
    
    # Indexes for optimal query performance
    __table_args__ = (
        Index('idx_wellness_time_astronaut', 'timestamp', 'astronaut_id'),
        Index('idx_wellness_cwi_time', 'cognitive_wellness_index', 'timestamp'),
        Index('idx_wellness_stress_time', 'stress_level', 'timestamp'),
        Index('idx_wellness_mission_time', 'mission_id', 'timestamp'),
    )

class WellnessAlert(Base):
    __tablename__ = 'wellness_alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    astronaut_id = Column(String(50), ForeignKey('astronauts.id'), nullable=False, index=True)
    mission_id = Column(Integer, ForeignKey('missions.id'), nullable=True)
    
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    priority = Column(Integer, default=1)
    
    # Alert trigger information
    trigger_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Alert content
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    recommended_actions = Column(Text, nullable=True)
    
    # Alert status
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True, index=True)  # Index for active alerts query
    resolved_by = Column(String(100), nullable=True)
    
    # Follow-up information
    follow_up_required = Column(Boolean, default=False)
    medical_review_required = Column(Boolean, default=False, index=True)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    astronaut = relationship("Astronaut", back_populates="alerts")
    mission = relationship("Mission", back_populates="alerts")

class WellnessReport(Base):
    __tablename__ = 'wellness_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    astronaut_id = Column(String(50), ForeignKey('astronauts.id'), nullable=False, index=True)
    mission_id = Column(Integer, ForeignKey('missions.id'), nullable=True)
    
    # Report metadata
    report_type = Column(String(50), nullable=False)
    report_period_start = Column(DateTime, nullable=False, index=True)
    report_period_end = Column(DateTime, nullable=False, index=True)
    generated_at = Column(DateTime, default=func.now(), nullable=False)
    generated_by = Column(String(100), nullable=False)
    
    # Pre-calculated summary statistics for fast report access
    avg_cwi_score = Column(Float, nullable=True)
    avg_stress_level = Column(Float, nullable=True)
    avg_fatigue_level = Column(Float, nullable=True)
    min_cwi_score = Column(Float, nullable=True)
    max_stress_level = Column(Float, nullable=True)
    max_fatigue_level = Column(Float, nullable=True)
    
    # Alert statistics
    total_alerts = Column(Integer, default=0)
    critical_alerts = Column(Integer, default=0)
    warning_alerts = Column(Integer, default=0)
    
    # Report status
    approval_status = Column(String(20), default='draft', index=True)
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    
    # File information
    file_path = Column(String(500), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    
    # Clinical notes
    clinical_summary = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    medical_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    astronaut = relationship("Astronaut", back_populates="reports")
    mission = relationship("Mission", back_populates="reports")

class SystemLog(Base):
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    level = Column(String(20), nullable=False, index=True)
    component = Column(String(50), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    astronaut_id = Column(String(50), nullable=True, index=True)
    mission_id = Column(Integer, nullable=True)

class EnhancedDatabaseManager:
    """PostgreSQL + TimescaleDB optimized database manager."""
    
    def __init__(self, database_url: str, enable_timescale: bool = False):  # CHANGED: Default to False
        self.database_url = database_url
        self.enable_timescale = enable_timescale
        
        # Connection pool settings for better performance
        pool_settings = {
            'pool_size': 10,  # REDUCED for compatibility
            'max_overflow': 20,  # REDUCED for compatibility
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'pool_pre_ping': True
        }
        
        self.engine = create_engine(database_url, **pool_settings, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Only create async engine if TimescaleDB is enabled
        if enable_timescale:
            async_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
            self.async_engine = create_async_engine(async_url, **pool_settings)
            self.AsyncSessionLocal = async_sessionmaker(self.async_engine, class_=AsyncSession)
        else:
            self.async_engine = None
            self.AsyncSessionLocal = None
        
    async def create_tables_and_timescale(self):
        """Create all database tables and set up TimescaleDB."""
        try:
            # Create regular tables first
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
            if self.enable_timescale and self.async_engine:
                await self._setup_timescaledb()
            else:
                logger.info("TimescaleDB setup skipped (disabled or not available)")
                
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    async def _setup_timescaledb(self):
        """Set up TimescaleDB hypertables and policies."""
        if not self.async_engine:
            logger.warning("Async engine not available for TimescaleDB setup")
            return
            
        async with self.async_engine.begin() as conn:
            try:
                # Create TimescaleDB extension
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                
                # Convert wellness_records to hypertable
                await conn.execute(text("""
                    SELECT create_hypertable('wellness_records', 'timestamp', 
                                            partitioning_column => 'astronaut_id',
                                            number_partitions => 4,
                                            if_not_exists => TRUE);
                """))
                
                logger.info("TimescaleDB setup completed successfully")
                
            except Exception as e:
                logger.error(f"TimescaleDB setup error: {e}")
                # Continue without TimescaleDB features
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    async def get_async_session(self):
        """Get async database session."""
        if self.AsyncSessionLocal:
            return self.AsyncSessionLocal()
        else:
            raise RuntimeError("Async session not available (TimescaleDB disabled)")

class OptimizedWellnessDataService:
    """High-performance wellness data service."""
    
    def __init__(self, db_manager: EnhancedDatabaseManager):
        self.db_manager = db_manager
    
    async def bulk_insert_wellness_data(self, data_batch: List[Dict[str, Any]]) -> List[int]:
        """Optimized bulk insert for high-frequency wellness data."""
        # Use sync session since TimescaleDB might not be available
        with self.db_manager.get_session() as session:
            records = [
                WellnessRecord(**data) for data in data_batch
            ]
            session.add_all(records)
            session.commit()
            return [record.id for record in records]  # FIXED: Use id instead of timestamp
    
    def get_report_data_optimized(self, astronaut_id: str, mission_id: Optional[int],
                                 start_time: datetime, end_time: datetime) -> Optional[Dict[str, Any]]:
        """Optimized report data generation."""
        with self.db_manager.get_session() as session:
            # Use regular PostgreSQL queries
            query = session.query(WellnessRecord).filter(
                WellnessRecord.astronaut_id == astronaut_id,
                WellnessRecord.timestamp >= start_time,
                WellnessRecord.timestamp <= end_time
            )
            
            if mission_id:
                query = query.filter(WellnessRecord.mission_id == mission_id)
            
            records = query.all()
            
            if not records:
                return None
            
            cwi_scores = [r.cognitive_wellness_index for r in records]
            stress_levels = [r.stress_level for r in records]
            fatigue_levels = [r.fatigue_level for r in records]
            
            return {
                'summary': {
                    'total_records': len(records),
                    'avg_cwi': sum(cwi_scores) / len(cwi_scores),
                    'min_cwi': min(cwi_scores),
                    'max_cwi': max(cwi_scores),
                    'avg_stress': sum(stress_levels) / len(stress_levels),
                    'max_stress': max(stress_levels),
                    'avg_fatigue': sum(fatigue_levels) / len(fatigue_levels),
                    'max_fatigue': max(fatigue_levels)
                },
                'records': records
            }

if __name__ == "__main__":
    # Test the fixed database model
    import asyncio
    from config import DATABASE_CONFIG
    
    async def test_database():
        db_manager = EnhancedDatabaseManager(DATABASE_CONFIG['url'], enable_timescale=False)
        
        try:
            await db_manager.create_tables_and_timescale()
            
            with db_manager.get_session() as session:
                # Test querying
                count = session.query(WellnessRecord).count()
                print(f"Current wellness records: {count}")
                
                # Test inserting
                test_record = WellnessRecord(
                    astronaut_id='CREW-001',
                    cognitive_wellness_index=80.0,
                    stress_level=30.0,
                    fatigue_level=25.0
                )
                
                session.add(test_record)
                session.commit()
                
                print(f"Test record inserted with ID: {test_record.id}")
                print("✅ Database model is working correctly!")
                
        except Exception as e:
            print(f"❌ Database test failed: {e}")
    
    asyncio.run(test_database())