#!/usr/bin/env python3
"""
Database setup script for MiVOLO Analytics System
This script creates the SQLite database and tables for storing analytics data.
"""

import os
import sys
import django
from django.conf import settings
from django.core.management import execute_from_command_line
from django.db import connection

def setup_django():
    """Configure Django settings"""
    if not settings.configured:
        settings.configure(
            DATABASES={
                'default': {
                    'ENGINE': 'django.db.backends.sqlite3',
                    'NAME': 'camera_analytics.db',
                }
            },
            INSTALLED_APPS=[
                'django.contrib.contenttypes',
                'django.contrib.auth',
            ],
            USE_TZ=True,
            TIME_ZONE='UTC',
            SECRET_KEY='your-secret-key-here',  # Required for Django
        )
        django.setup()

def create_tables():
    """Create database tables"""
    print("Creating database tables...")
    
    # Import models after Django setup
    from django_models import CameraSession, AnalyticsFrame, PersonDetection, FaceDetection, SessionSummary
    
    # Create tables using raw SQL (since we don't have migrations)
    with connection.cursor() as cursor:
        # Create camera_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_sessions (
                session_id TEXT PRIMARY KEY,
                camera_name VARCHAR(100) NOT NULL,
                location VARCHAR(200) NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
        ''')
        
        # Create analytics_frames table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_frames (
                frame_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                frame_number INTEGER NOT NULL,
                total_persons INTEGER NOT NULL DEFAULT 0,
                total_faces INTEGER NOT NULL DEFAULT 0,
                male_count INTEGER NOT NULL DEFAULT 0,
                female_count INTEGER NOT NULL DEFAULT 0,
                unknown_gender_count INTEGER NOT NULL DEFAULT 0,
                children_count INTEGER NOT NULL DEFAULT 0,
                young_adults_count INTEGER NOT NULL DEFAULT 0,
                middle_aged_count INTEGER NOT NULL DEFAULT 0,
                seniors_count INTEGER NOT NULL DEFAULT 0,
                unknown_age_count INTEGER NOT NULL DEFAULT 0,
                processing_time_ms REAL NOT NULL,
                confidence_threshold REAL NOT NULL DEFAULT 0.5,
                FOREIGN KEY (session_id) REFERENCES camera_sessions (session_id)
            )
        ''')
        
        # Create person_detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_detections (
                detection_id TEXT PRIMARY KEY,
                frame_id TEXT NOT NULL,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_width REAL NOT NULL,
                bbox_height REAL NOT NULL,
                predicted_age INTEGER,
                age_group VARCHAR(10) NOT NULL DEFAULT 'UNKNOWN',
                predicted_gender VARCHAR(1) NOT NULL DEFAULT 'U',
                person_confidence REAL NOT NULL,
                age_confidence REAL,
                gender_confidence REAL,
                track_id INTEGER,
                FOREIGN KEY (frame_id) REFERENCES analytics_frames (frame_id)
            )
        ''')
        
        # Create face_detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_detections (
                detection_id TEXT PRIMARY KEY,
                frame_id TEXT NOT NULL,
                person_detection_id TEXT,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_width REAL NOT NULL,
                bbox_height REAL NOT NULL,
                predicted_age INTEGER,
                predicted_gender VARCHAR(1) NOT NULL DEFAULT 'U',
                face_confidence REAL NOT NULL,
                age_confidence REAL,
                gender_confidence REAL,
                FOREIGN KEY (frame_id) REFERENCES analytics_frames (frame_id),
                FOREIGN KEY (person_detection_id) REFERENCES person_detections (detection_id)
            )
        ''')
        
        # Create session_summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                total_frames_processed INTEGER NOT NULL DEFAULT 0,
                unique_persons_detected INTEGER NOT NULL DEFAULT 0,
                total_person_detections INTEGER NOT NULL DEFAULT 0,
                total_face_detections INTEGER NOT NULL DEFAULT 0,
                total_males INTEGER NOT NULL DEFAULT 0,
                total_females INTEGER NOT NULL DEFAULT 0,
                total_unknown_gender INTEGER NOT NULL DEFAULT 0,
                total_children INTEGER NOT NULL DEFAULT 0,
                total_young_adults INTEGER NOT NULL DEFAULT 0,
                total_middle_aged INTEGER NOT NULL DEFAULT 0,
                total_seniors INTEGER NOT NULL DEFAULT 0,
                total_unknown_age INTEGER NOT NULL DEFAULT 0,
                avg_processing_time_ms REAL NOT NULL DEFAULT 0.0,
                max_persons_in_frame INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                FOREIGN KEY (session_id) REFERENCES camera_sessions (session_id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_frames_session_timestamp ON analytics_frames (session_id, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_frames_timestamp ON analytics_frames (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_detections_frame_track ON person_detections (frame_id, track_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_detections_gender ON person_detections (predicted_gender)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_detections_age_group ON person_detections (age_group)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_detections_frame ON face_detections (frame_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_detections_gender ON face_detections (predicted_gender)')
    
    print("Database tables created successfully!")

def create_sample_data():
    """Create some sample data for testing"""
    print("Creating sample data...")
    
    from django_models import CameraSession, AnalyticsFrame, PersonDetection, FaceDetection
    from django.utils import timezone
    import uuid
    
    # Create a sample camera session
    session = CameraSession.objects.create(
        camera_name="Test Camera 1",
        location="Main Entrance",
        start_time=timezone.now()
    )
    
    # Create a sample analytics frame
    frame = AnalyticsFrame.objects.create(
        session=session,
        frame_number=1,
        total_persons=2,
        total_faces=2,
        male_count=1,
        female_count=1,
        young_adults_count=2,
        processing_time_ms=150.5
    )
    
    # Create sample person detections
    PersonDetection.objects.create(
        frame=frame,
        bbox_x=0.1,
        bbox_y=0.2,
        bbox_width=0.3,
        bbox_height=0.4,
        predicted_age=25,
        predicted_gender='M',
        person_confidence=0.95
    )
    
    PersonDetection.objects.create(
        frame=frame,
        bbox_x=0.5,
        bbox_y=0.3,
        bbox_width=0.25,
        bbox_height=0.35,
        predicted_age=28,
        predicted_gender='F',
        person_confidence=0.92
    )
    
    print("Sample data created successfully!")

def show_database_info():
    """Show information about the created database"""
    print("\n" + "="*50)
    print("DATABASE SETUP COMPLETE")
    print("="*50)
    print(f"Database file: camera_analytics.db")
    print(f"Location: {os.path.abspath('camera_analytics.db')}")
    
    # Show table information
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\nCreated tables:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  - {table_name}: {count} records")
    
    print("\nYou can now run:")
    print("  1. python new_server.py  (to start the analytics server)")
    print("  2. python new_client.py  (to start the camera client)")
    print("\nDatabase can be viewed with any SQLite browser or:")
    print("  sqlite3 camera_analytics.db")

def main():
    """Main setup function"""
    print("Setting up MiVOLO Analytics Database...")
    print("-" * 40)
    
    # Setup Django
    setup_django()
    
    # Create tables
    create_tables()
    
    # Ask if user wants sample data
    create_sample = input("\nCreate sample data for testing? (y/n): ").lower().strip()
    if create_sample in ['y', 'yes']:
        create_sample_data()
    
    # Show database info
    show_database_info()

if __name__ == "__main__":
    main()
