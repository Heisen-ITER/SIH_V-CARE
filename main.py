# main_enhanced.py - Updated for 1-minute database saves and full data reports

import asyncio
import socketio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
import cv2
import logging
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import datetime
import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from pathlib import Path
import io

# Import your custom modules
try:
    import mavni_1_video as mavni 
    import vani
    import engine
    from database_model import (
        EnhancedDatabaseManager,
        OptimizedWellnessDataService,
        Astronaut, 
        Mission, 
        WellnessRecord, 
        WellnessAlert
    )
    from medical_report_generator import ClinicalReportGenerator
    from config import DATABASE_CONFIG, VIDEO_CONFIG, PATHS_CONFIG
except ImportError as e:
    print(f"FATAL ERROR: A core module is missing. Details: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---
class AlertResponse(BaseModel):
    id: int
    astronaut_id: str
    alert_type: str
    severity: str
    timestamp: datetime.datetime
    description: str
    acknowledged: bool

class ReportRequest(BaseModel):
    astronaut_id: str
    mission_id: Optional[int] = None
    start_time: datetime.datetime
    end_time: datetime.datetime
    report_type: str = "routine"

class SystemStatus(BaseModel):
    system_online: bool
    database_connected: bool
    video_active: bool
    audio_active: bool
    active_alerts: int
    connected_clients: int

# --- Global Variables ---
cap = None
data_task = None
connected_clients = set()
executor = ThreadPoolExecutor(max_workers=3)
db_manager = None
data_service = None
current_mission = None
last_alert_times = {}
report_generator = None

# NEW: Store accumulated data for batch saving
accumulated_wellness_data = []
data_accumulation_lock = asyncio.Lock()

# Ensure directories exist
for dir_path in [PATHS_CONFIG['reports_dir'], PATHS_CONFIG['static_files_dir']]:
    Path(dir_path).mkdir(exist_ok=True)

def try_video_paths():
    """Try video paths from config in order."""
    for video_path in VIDEO_CONFIG['video_paths']:
        if isinstance(video_path, int):  # Webcam
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                logger.info(f"Using webcam: {video_path}")
                return cap
            cap.release()
        else:  # File path
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    logger.info(f"Using video file: {video_path}")
                    return cap
                cap.release()
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced startup and shutdown with proper error handling."""
    global cap, db_manager, data_service, current_mission, report_generator
    
    try:
        # Initialize database with better error handling
        logger.info("Initializing database connection...")
        try:
            db_manager = EnhancedDatabaseManager(DATABASE_CONFIG['url'])
            await db_manager.create_tables_and_timescale()
            data_service = OptimizedWellnessDataService(db_manager)
            logger.info("Database initialized successfully.")
        except Exception as db_error:
            logger.error(f"Database initialization failed: {db_error}")
            # Continue without database - use mock data service
            db_manager = None
            data_service = MockWellnessDataService()
            logger.warning("Running in mock mode without database")
        
        # Initialize report generator
        report_generator = ClinicalReportGenerator(PATHS_CONFIG['reports_dir'])
        
        # Initialize or get current mission
        if db_manager:
            current_mission = await initialize_mission()
        
        # Initialize video capture with fallback
        logger.info("Initializing video capture...")
        cap = try_video_paths()
        if cap:
            logger.info("Video capture initialized successfully.")
        else:
            logger.warning("No video source available. System will run without video feed.")
        
        # Start background tasks
        if db_manager:
            asyncio.create_task(cleanup_old_data())
            # NEW: Start the 1-minute database save task
            asyncio.create_task(periodic_database_save())
        
    except Exception as e:
        logger.error(f"FATAL ERROR during server startup: {e}")
        # Don't exit - allow system to run in degraded mode
    
    yield
    
    # Shutdown cleanup
    logger.info("Server shutting down...")
    # Save any remaining accumulated data
    if db_manager and accumulated_wellness_data:
        try:
            await save_accumulated_data_to_db()
        except Exception as e:
            logger.error(f"Error saving final data: {e}")
    
    if cap and cap.isOpened(): 
        cap.release()
    if db_manager:
        # Close database connections if needed
        pass
    executor.shutdown(wait=True)
    logger.info("All resources released.")

class MockWellnessDataService:
    """Mock service when database is unavailable."""
    
    def get_report_data_optimized(self, astronaut_id: str, mission_id: Optional[int],
                                 start_time: datetime.datetime, end_time: datetime.datetime) -> Optional[Dict[str, Any]]:
        """Return mock data for report generation."""
        return {
            'summary': {
                'avg_cwi': 75.5,
                'avg_stress': 25.3,
                'avg_fatigue': 20.1,
                'min_cwi': 65.0,
                'max_stress': 45.2,
                'max_fatigue': 35.8,
                'total_records': 500
            }
        }

async def initialize_mission():
    """Initialize or retrieve current mission."""
    try:
        with db_manager.get_session() as session:
            # Check if astronaut exists, create if not
            astronaut = session.query(Astronaut).filter_by(id="CREW-001").first()
            if not astronaut:
                astronaut = Astronaut(
                    id="CREW-001",
                    full_name="Test Astronaut",
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
                session.commit()
                logger.info("Created new astronaut profile: CREW-001")
            
            # Check for active mission
            active_mission = session.query(Mission).filter(
                Mission.astronaut_id == "CREW-001",
                Mission.end_date.is_(None)
            ).first()
            
            if not active_mission:
                active_mission = Mission(
                    mission_code=f"DEMO-MISSION-{datetime.datetime.now().strftime('%Y%m%d')}",
                    astronaut_id="CREW-001",
                    mission_type="Simulation",
                    start_date=datetime.datetime.utcnow(),
                    current_phase="orbit",
                    duration_hours=24,
                    mission_objectives="Real-time cognitive wellness monitoring demonstration",
                    risk_level="moderate"
                )
                session.add(active_mission)
                session.commit()
                logger.info(f"Created new mission: {active_mission.mission_code}")
            
            return active_mission
            
    except Exception as e:
        logger.error(f"Error initializing mission: {e}")
        return None

app = FastAPI(lifespan=lifespan, title="V-CARE Cognitive Wellness API", version="2.0.0")
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
app.mount('/socket.io', socketio.ASGIApp(sio))

# Serve static files
app.mount("/static", StaticFiles(directory=PATHS_CONFIG['static_files_dir']), name="static")

# NEW: Periodic database save function (every 1 minute)
async def periodic_database_save():
    """Save accumulated wellness data to database every 1 minute."""
    while True:
        try:
            await asyncio.sleep(60)  # Wait 1 minute
            await save_accumulated_data_to_db()
        except Exception as e:
            logger.error(f"Error in periodic database save: {e}")

async def save_accumulated_data_to_db():
    """Save all accumulated wellness data to database."""
    if not db_manager or not accumulated_wellness_data:
        return
    
    async with data_accumulation_lock:
        if not accumulated_wellness_data:
            return
        
        try:
            # Save all accumulated data
            await data_service.bulk_insert_wellness_data(accumulated_wellness_data)
            logger.info(f"Saved {len(accumulated_wellness_data)} wellness records to database")
            
            # Clear the accumulated data
            accumulated_wellness_data.clear()
            
        except Exception as e:
            logger.error(f"Error saving accumulated data: {e}")

async def send_data_updates():
    """Enhanced data update loop - now accumulates data for 1-minute saves."""
    if not cap:
        logger.error("No video capture available. Cannot start data updates.")
        return
        
    logger.info("Starting enhanced data update task with 1-minute database saves...")
    loop = asyncio.get_running_loop()

    # Initialize mavni's global state
    mavni.last_analysis_result = {
        "primary_emotion": "---",
        "raw_emotion_scores": np.zeros((1, len(mavni.EMOTION_LABELS))),
        "face_crop_for_debug": None
    }

    frame_counter = 0
    FRAME_SKIP = VIDEO_CONFIG.get('frame_skip', 2)

    while connected_clients and cap and cap.isOpened():
        try:
            ret, frame = cap.read()
            
            if not ret:
                logger.info("Video ended. Looping from the beginning.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_counter += 1
            if frame_counter % FRAME_SKIP != 0:
                await asyncio.sleep(1/60)
                continue

            # Get analysis data
            mavni_task = loop.run_in_executor(executor, mavni.analyze_frame, frame)
            vani_data = {"speech_anomaly_score": 0.0} 
            mavni_data = await mavni_task
            fused_data = engine.fuse_data(mavni_data, vani_data)
            
            # NEW: Accumulate data for batch saving every minute
            if db_manager and current_mission:
                await accumulate_wellness_data(fused_data, mavni_data, vani_data)
                await check_and_create_alerts(fused_data)
            
            # Send to connected dashboards
            await sio.emit('update_data', fused_data)
            
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in enhanced data update loop: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(1)
            
    logger.info("Enhanced data update task stopped.")

async def accumulate_wellness_data(fused_data: Dict[str, Any], mavni_data: Dict[str, Any], vani_data: Dict[str, Any]):
    """Accumulate wellness data for batch saving every minute."""
    if not db_manager or not current_mission:
        return
        
    try:
        wellness_data = {
            'timestamp': datetime.datetime.utcnow(),
            'astronaut_id': 'CREW-001',
            'mission_id': current_mission.id,
            'cognitive_wellness_index': fused_data.get('cognitive_wellness_index', 0),
            'stress_level': fused_data.get('stress_level', 0),
            'fatigue_level': fused_data.get('fatigue_level', 0),
            'primary_emotion': fused_data.get('factors', {}).get('emotion', 'Unknown'),
            'emotion_confidence': 0.8,
            'blink_rate': fused_data.get('factors', {}).get('blink_rate', 0),
            'vocal_pitch': vani_data.get('vocal_pitch', 0),
            'vocal_energy': vani_data.get('vocal_energy', 0),
            'vocal_anomaly_score': fused_data.get('factors', {}).get('vocal_anomaly_factor', 0),
            'mission_phase': current_mission.current_phase,
            'activity_type': 'monitoring',
            'workload_level': 'moderate'
        }
        
        # Add to accumulated data
        async with data_accumulation_lock:
            accumulated_wellness_data.append(wellness_data)
            
        logger.debug(f"Accumulated wellness data (total: {len(accumulated_wellness_data)})")
        
    except Exception as e:
        logger.error(f"Error accumulating wellness data: {e}")

async def check_and_create_alerts(fused_data: Dict[str, Any]):
    """Check for alert conditions and create alerts."""
    if not db_manager:
        return
        
    try:
        now = datetime.datetime.utcnow()
        stress_level = fused_data.get('stress_level', 0)
        fatigue_level = fused_data.get('fatigue_level', 0)
        cwi_score = fused_data.get('cognitive_wellness_index', 100)
        
        # High stress alert
        if stress_level > 75:
            alert_key = f"stress_{stress_level//10}0"
            if should_create_alert(alert_key):
                await create_alert({
                    'astronaut_id': 'CREW-001',
                    'mission_id': current_mission.id if current_mission else None,
                    'alert_type': 'high_stress',
                    'severity': 'critical' if stress_level > 90 else 'warning',
                    'trigger_value': stress_level,
                    'threshold_value': 75,
                    'title': f'High Stress Alert - {stress_level}%',
                    'description': f'Elevated stress level detected: {stress_level}%',
                    'recommended_actions': 'Monitor crew member closely, consider stress reduction protocols'
                })
        
        # High fatigue alert
        if fatigue_level > 80:
            alert_key = f"fatigue_{fatigue_level//10}0"
            if should_create_alert(alert_key):
                await create_alert({
                    'astronaut_id': 'CREW-001',
                    'mission_id': current_mission.id if current_mission else None,
                    'alert_type': 'high_fatigue',
                    'severity': 'critical' if fatigue_level > 90 else 'warning',
                    'trigger_value': fatigue_level,
                    'threshold_value': 80,
                    'title': f'High Fatigue Alert - {fatigue_level}%',
                    'description': f'Elevated fatigue level detected: {fatigue_level}%',
                    'recommended_actions': 'Implement rest protocols, evaluate workload'
                })
        
        # Low CWI alert
        if cwi_score < 40:
            alert_key = f"low_cwi_{cwi_score//10}0"
            if should_create_alert(alert_key):
                await create_alert({
                    'astronaut_id': 'CREW-001',
                    'mission_id': current_mission.id if current_mission else None,
                    'alert_type': 'low_cwi',
                    'severity': 'critical' if cwi_score < 25 else 'warning',
                    'trigger_value': cwi_score,
                    'threshold_value': 40,
                    'title': f'Low CWI Alert - {cwi_score}%',
                    'description': f'Low cognitive wellness index: {cwi_score}%',
                    'recommended_actions': 'Immediate medical evaluation recommended'
                })
        
    except Exception as e:
        logger.error(f"Error checking alerts: {e}")

def should_create_alert(alert_key: str) -> bool:
    """Check if enough time has passed to create a new alert of this type."""
    now = datetime.datetime.utcnow()
    if alert_key in last_alert_times:
        time_since_last = now - last_alert_times[alert_key]
        if time_since_last.total_seconds() < 300:  # 5 minutes cooldown
            return False
    
    last_alert_times[alert_key] = now
    return True

async def create_alert(alert_data: Dict[str, Any]):
    """Create a new wellness alert."""
    if not db_manager:
        return
        
    try:
        with db_manager.get_session() as session:
            alert = WellnessAlert(**alert_data)
            session.add(alert)
            session.commit()
            alert_id = alert.id
            
        logger.warning(f"Created alert {alert_id}: {alert_data['description']}")
        
        # Emit alert to connected clients
        await sio.emit('wellness_alert', {
            'id': alert_id,
            'type': alert_data['alert_type'],
            'severity': alert_data['severity'],
            'description': alert_data['description'],
            'timestamp': datetime.datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error creating alert: {e}")

async def cleanup_old_data():
    """Background task to clean up old data."""
    if not db_manager:
        return
        
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(hours=72)
            
            with db_manager.get_session() as session:
                old_records = session.query(WellnessRecord).filter(
                    WellnessRecord.timestamp < cutoff_time
                ).count()
                
                if old_records > 0:
                    session.query(WellnessRecord).filter(
                        WellnessRecord.timestamp < cutoff_time
                    ).delete()
                    session.commit()
                    logger.info(f"Cleaned up {old_records} old wellness records")
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

# --- Helper functions to get ALL data from database ---
def get_real_astronaut_data():
    """Get real astronaut data from database."""
    if not db_manager:
        # Fallback to basic data if no database
        return {
            'id': 'CREW-001',
            'full_name': 'Test Astronaut',
            'baseline_cwi_score': 85.0,
            'baseline_stress_threshold': 70.0,
            'baseline_fatigue_threshold': 65.0
        }
    
    try:
        with db_manager.get_session() as session:
            astronaut = session.query(Astronaut).filter_by(id="CREW-001").first()
            if astronaut:
                return {
                    'id': astronaut.id,
                    'full_name': astronaut.full_name,
                    'baseline_cwi_score': astronaut.baseline_cwi_score,
                    'baseline_stress_threshold': astronaut.baseline_stress_threshold,
                    'baseline_fatigue_threshold': astronaut.baseline_fatigue_threshold,
                    'date_of_birth': astronaut.date_of_birth.isoformat() if astronaut.date_of_birth else None,
                    'medical_clearance_date': astronaut.medical_clearance_date.isoformat() if astronaut.medical_clearance_date else None,
                    'psychological_profile': astronaut.psychological_profile or {}
                }
    except Exception as e:
        logger.error(f"Error getting astronaut data: {e}")
    
    # Fallback
    return {
        'id': 'CREW-001',
        'full_name': 'Test Astronaut',
        'baseline_cwi_score': 85.0,
        'baseline_stress_threshold': 70.0,
        'baseline_fatigue_threshold': 65.0
    }

def get_real_mission_data():
    """Get real mission data from database."""
    if not db_manager or not current_mission:
        return {
            'mission_code': 'DEMO-MISSION',
            'mission_type': 'Simulation',
            'current_phase': 'monitoring',
            'risk_level': 'moderate'
        }
    
    try:
        return {
            'mission_code': current_mission.mission_code,
            'mission_type': current_mission.mission_type,
            'current_phase': current_mission.current_phase,
            'risk_level': current_mission.risk_level,
            'start_date': current_mission.start_date.isoformat() if current_mission.start_date else None,
            'end_date': current_mission.end_date.isoformat() if current_mission.end_date else None,
            'duration_hours': current_mission.duration_hours,
            'mission_objectives': current_mission.mission_objectives
        }
    except Exception as e:
        logger.error(f"Error getting mission data: {e}")
        return {
            'mission_code': 'DEMO-MISSION',
            'mission_type': 'Simulation',
            'current_phase': 'monitoring',
            'risk_level': 'moderate'
        }

# NEW: Get ALL wellness data from database (not just last 24 hours)
def get_all_wellness_timeline():
    """Get ALL wellness timeline data from database."""
    if not db_manager:
        logger.warning("No database manager, using empty wellness timeline")
        return []
    
    try:
        with db_manager.get_session() as session:
            # Get ALL wellness records for this astronaut (no time limit)
            wellness_records = session.query(WellnessRecord).filter(
                WellnessRecord.astronaut_id == 'CREW-001'
            ).order_by(WellnessRecord.timestamp).all()
            
            timeline = []
            for record in wellness_records:
                timeline.append({
                    'timestamp': record.timestamp.isoformat(),
                    'cognitive_wellness_index': record.cognitive_wellness_index or 0,
                    'stress_level': record.stress_level or 0,
                    'fatigue_level': record.fatigue_level or 0,
                    'primary_emotion': record.primary_emotion or 'Unknown',
                    'emotion_confidence': record.emotion_confidence or 0,
                    'blink_rate': record.blink_rate or 0,
                    'eye_closure_duration': record.eye_closure_duration or 0,
                    'vocal_pitch': record.vocal_pitch or 0,
                    'vocal_energy': record.vocal_energy or 0,
                    'vocal_anomaly_score': record.vocal_anomaly_score or 0,
                    'mission_phase': record.mission_phase or 'unknown',
                    'activity_type': record.activity_type or 'monitoring',
                    'workload_level': record.workload_level or 'normal'
                })
            
            logger.info(f"Retrieved {len(timeline)} ALL wellness records from database")
            return timeline
            
    except Exception as e:
        logger.error(f"Error getting all wellness timeline: {e}")
        return []

# NEW: Get ALL alerts from database
def get_all_alerts():
    """Get ALL alerts from database."""
    if not db_manager:
        logger.warning("No database manager, using empty alerts")
        return []
    
    try:
        with db_manager.get_session() as session:
            # Get ALL alert records for this astronaut (no time limit)
            alert_records = session.query(WellnessAlert).filter(
                WellnessAlert.astronaut_id == 'CREW-001'
            ).order_by(WellnessAlert.timestamp).all()
            
            alerts = []
            for alert in alert_records:
                alerts.append({
                    'timestamp': alert.timestamp.isoformat(),
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'title': alert.title,
                    'description': alert.description,
                    'trigger_value': alert.trigger_value,
                    'threshold_value': alert.threshold_value,
                    'recommended_actions': alert.recommended_actions,
                    'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
                })
            
            logger.info(f"Retrieved {len(alerts)} ALL alerts from database")
            return alerts
            
    except Exception as e:
        logger.error(f"Error getting all alerts: {e}")
        return []

# --- API Routes ---

@app.get("/")
def read_root():
    return {"status": "V-CARE Cognitive Wellness Backend is running", "version": "2.0.0"}

@app.get("/dashboard")
def serve_dashboard():
    """Serve the main dashboard."""
    dashboard_path = Path(__file__).parent / "new_dashboard.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path))
    return {"error": "Dashboard not found"}

@app.get("/generate-report")
async def generate_medical_report():
    """Generate and return a medical report PDF with ALL stored data."""
    try:
        if not report_generator:
            raise HTTPException(status_code=503, detail="Report generator not available")
        
        logger.info("Generating medical report with ALL stored database data...")
        
        # Save any accumulated data first
        if accumulated_wellness_data:
            await save_accumulated_data_to_db()
        
        # Get ALL data from database (not time-limited)
        astronaut_data = get_real_astronaut_data()
        mission_data = get_real_mission_data()
        wellness_timeline = get_all_wellness_timeline()  # NEW: Get ALL data
        alerts = get_all_alerts()  # NEW: Get ALL alerts
        
        logger.info(f"Retrieved {len(wellness_timeline)} ALL wellness records and {len(alerts)} ALL alerts for report")
        
        # Determine report period from the actual data
        if wellness_timeline:
            start_time = datetime.datetime.fromisoformat(wellness_timeline[0]['timestamp'])
            end_time = datetime.datetime.fromisoformat(wellness_timeline[-1]['timestamp'])
        else:
            end_time = datetime.datetime.utcnow()
            start_time = end_time - datetime.timedelta(hours=1)
        
        # Generate the report with ALL data
        report_path = report_generator.generate_clinical_report(
            astronaut_data=astronaut_data,
            mission_data=mission_data,
            wellness_timeline=wellness_timeline,
            alerts=alerts,
            report_period=(start_time, end_time)
        )
        
        # Return the PDF as a streaming response
        def iterfile():
            with open(report_path, mode="rb") as file_like:
                yield from file_like
        
        headers = {
            'Content-Disposition': f'attachment; filename="V-CARE_Complete_Report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
        }
        
        return StreamingResponse(iterfile(), media_type="application/pdf", headers=headers)
        
    except Exception as e:
        logger.error(f"Error generating medical report: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/api/generate-report-download")
async def generate_report_download():
    """Generate and return a medical report PDF with ALL stored data."""
    try:
        if not report_generator:
            raise HTTPException(status_code=503, detail="Report generator not available")
        
        logger.info("Starting complete report generation with ALL stored data...")
        
        # Save any accumulated data first
        if accumulated_wellness_data:
            await save_accumulated_data_to_db()
        
        # Get ALL data from database
        astronaut_data = get_real_astronaut_data()
        mission_data = get_real_mission_data()
        wellness_timeline = get_all_wellness_timeline()  # ALL DATA
        alerts = get_all_alerts()  # ALL ALERTS
        
        logger.info(f"Retrieved {len(wellness_timeline)} ALL wellness records and {len(alerts)} ALL alerts")
        
        # Determine report period from actual data
        if wellness_timeline:
            start_time = datetime.datetime.fromisoformat(wellness_timeline[0]['timestamp'])
            end_time = datetime.datetime.fromisoformat(wellness_timeline[-1]['timestamp'])
            logger.info(f"Report period: {start_time} to {end_time}")
        else:
            end_time = datetime.datetime.utcnow()
            start_time = end_time - datetime.timedelta(hours=1)
            logger.warning("No data found, using 1-hour default period")
        
        # Generate the report with ALL data
        report_path = report_generator.generate_clinical_report(
            astronaut_data=astronaut_data,
            mission_data=mission_data,
            wellness_timeline=wellness_timeline,
            alerts=alerts,
            report_period=(start_time, end_time)
        )
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=500, detail="Report file was not created")
        
        # Create proper filename
        filename = f"V-CARE_Complete_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        logger.info(f"Report generated successfully with ALL stored data: {report_path}")
        
        # Read the file and return as response with proper headers
        try:
            with open(report_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            raise HTTPException(status_code=500, detail="Error reading generated report")
        
        if len(pdf_content) == 0:
            raise HTTPException(status_code=500, detail="Generated PDF is empty")
        
        logger.info(f"Returning PDF with ALL stored data, size: {len(pdf_content)} bytes")
        
        # Create response with proper headers for download
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf",
                "Content-Length": str(len(pdf_content)),
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating medical report with ALL stored data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/api/system/status", response_model=SystemStatus)
def get_system_status():
    """Get current system status."""
    active_alerts = 0
    try:
        if db_manager:
            with db_manager.get_session() as session:
                active_alerts = session.query(WellnessAlert).filter(
                    WellnessAlert.resolved_at.is_(None)
                ).count()
    except:
        pass
    
    return SystemStatus(
        system_online=True,
        database_connected=db_manager is not None,
        video_active=cap is not None and cap.isOpened(),
        audio_active=False,  # Disabled for video mode
        active_alerts=active_alerts,
        connected_clients=len(connected_clients)
    )

@app.get("/api/reports/list")
def list_reports():
    """List all available reports."""
    reports_dir = Path(PATHS_CONFIG['reports_dir'])
    reports = []
    
    if reports_dir.exists():
        for file_path in reports_dir.glob("*.pdf"):
            stat = file_path.stat()
            reports.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "full_path": str(file_path.absolute())
            })
    
    # Sort by creation time, newest first
    reports.sort(key=lambda x: x["created"], reverse=True)
    
    return {
        "reports_directory": str(reports_dir.absolute()),
        "total_reports": len(reports),
        "reports": reports
    }

@app.get("/api/reports/download/{filename}")
def download_report(filename: str):
    """Download a specific report by filename."""
    reports_dir = Path(PATHS_CONFIG['reports_dir'])
    file_path = reports_dir / filename
    
    # Security check - ensure filename is safe and file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    
    if not filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        str(file_path),
        media_type="application/pdf",
        filename=filename
    )

# NEW: API endpoint to get database statistics
@app.get("/api/database/stats")
def get_database_stats():
    """Get database statistics."""
    if not db_manager:
        return {"error": "Database not available"}
    
    try:
        with db_manager.get_session() as session:
            total_wellness_records = session.query(WellnessRecord).count()
            total_alerts = session.query(WellnessAlert).count()
            active_alerts = session.query(WellnessAlert).filter(
                WellnessAlert.resolved_at.is_(None)
            ).count()
            
            # Get latest record timestamp
            latest_record = session.query(WellnessRecord).order_by(
                WellnessRecord.timestamp.desc()
            ).first()
            
            latest_timestamp = latest_record.timestamp.isoformat() if latest_record else None
            
            # Get oldest record timestamp
            oldest_record = session.query(WellnessRecord).order_by(
                WellnessRecord.timestamp.asc()
            ).first()
            
            oldest_timestamp = oldest_record.timestamp.isoformat() if oldest_record else None
            
            return {
                "total_wellness_records": total_wellness_records,
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "accumulated_records_pending": len(accumulated_wellness_data),
                "latest_record": latest_timestamp,
                "oldest_record": oldest_timestamp,
                "database_connected": True
            }
            
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e), "database_connected": False}

# NEW: API endpoint to force save accumulated data
@app.post("/api/database/force-save")
async def force_save_accumulated_data():
    """Force save any accumulated data to database."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        if not accumulated_wellness_data:
            return {"message": "No accumulated data to save", "records_saved": 0}
        
        records_count = len(accumulated_wellness_data)
        await save_accumulated_data_to_db()
        
        return {
            "message": "Accumulated data saved successfully",
            "records_saved": records_count,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error force saving data: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")

# --- Socket.IO Events ---

@sio.event
async def connect(sid, environ):
    global data_task
    logger.info(f"Client connected: {sid}")
    connected_clients.add(sid)
    
    # Send initial system status with database info
    status_data = {
        'connected': True,
        'mission': current_mission.mission_code if current_mission else None,
        'astronaut': 'CREW-001',
        'database_connected': db_manager is not None
    }
    
    # Add database statistics if available
    if db_manager:
        try:
            with db_manager.get_session() as session:
                total_records = session.query(WellnessRecord).count()
                status_data['total_records'] = total_records
                status_data['accumulated_pending'] = len(accumulated_wellness_data)
        except:
            pass
    
    await sio.emit('system_status', status_data, room=sid)
    
    if not cap:
        logger.warning("Video file not available. Data stream will not start.")
        return
    
    if data_task is None or data_task.done():
        data_task = asyncio.create_task(send_data_updates())

@sio.event
def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    connected_clients.discard(sid)

# --- Development Server ---
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print(" V-CARE COGNITIVE WELLNESS SYSTEM ".center(60, "="))
    print("="*60)
    print(f"Dashboard: http://localhost:8000/dashboard")
    print(f"Reports:   http://localhost:8000/generate-report") 
    print(f"Direct DL: http://localhost:8000/api/generate-report-download")
    print(f"API Docs:  http://localhost:8000/docs")
    print(f"Status:    http://localhost:8000/api/system/status")
    print(f"DB Stats:  http://localhost:8000/api/database/stats")
    print("="*60)
    print("NEW FEATURES:")
    print("• Data saved to database every 1 minute")
    print("• Reports include ALL stored database records")
    print("• Force save API: POST /api/database/force-save")
    print("• Database statistics API available")
    print("="*60)
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )