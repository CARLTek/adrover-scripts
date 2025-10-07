#!/usr/bin/env python3
"""
MiVOLO Analytics Data Viewer
View stored analytics data from JSON files
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

def load_session_data(session_file: str) -> Dict:
    """Load session data from JSON file"""
    try:
        with open(session_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {session_file}: {e}")
        return None

def format_duration(start_time: str, end_time: str = None) -> str:
    """Format duration between start and end time"""
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        else:
            end = datetime.now()
        
        duration = end - start
        total_seconds = int(duration.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        return f"{minutes}m {seconds}s"
    except:
        return "Unknown"

def print_session_summary(session_data: Dict):
    """Print a summary of the session"""
    print("=" * 60)
    print("📊 MiVOLO ANALYTICS SESSION SUMMARY")
    print("=" * 60)
    
    # Basic info
    print(f"🆔 Session ID: {session_data['session_id']}")
    print(f"⏰ Start Time: {session_data['start_time']}")
    
    # Get end time from last frame
    end_time = None
    if session_data['frames']:
        end_time = session_data['frames'][-1]['timestamp']
    
    duration = format_duration(session_data['start_time'], end_time)
    print(f"⏱️  Duration: {duration}")
    
    summary = session_data['summary']
    print(f"🎬 Total Frames: {summary['total_frames']}")

    # Show unique persons (corrected count)
    if 'estimated_unique_persons' in summary:
        print(f"👥 Unique Persons: {summary['estimated_unique_persons']}")
        print(f"😊 Unique Faces: {summary['estimated_unique_faces']}")
        print(f"📊 Total Detections: {summary.get('total_person_detections', 0)} person detections across all frames")
    else:
        # Fallback for old format
        print(f"👥 Total Person Detections: {summary.get('total_persons_detected', 0)}")
        print(f"😊 Total Face Detections: {summary.get('total_faces_detected', 0)}")
        print("⚠️  Note: This shows detections per frame, not unique individuals")
    
    print("\n" + "=" * 30 + " DEMOGRAPHICS " + "=" * 30)
    
    # Gender distribution (use current demographics if available)
    if 'current_demographics' in summary:
        demo = summary['current_demographics']
        print("👫 Current Demographics (Unique Individuals):")
        total_persons = demo['male_count'] + demo['female_count'] + demo['unknown_gender_count']
        if total_persons > 0:
            if demo['male_count'] > 0:
                print(f"   Male: {demo['male_count']} ({(demo['male_count']/total_persons)*100:.1f}%)")
            if demo['female_count'] > 0:
                print(f"   Female: {demo['female_count']} ({(demo['female_count']/total_persons)*100:.1f}%)")
            if demo['unknown_gender_count'] > 0:
                print(f"   Unknown: {demo['unknown_gender_count']} ({(demo['unknown_gender_count']/total_persons)*100:.1f}%)")

        print("\n🎂 Age Group Distribution (Unique Individuals):")
        age_labels = {
            'children_count': '👶 Children (0-17)',
            'young_adults_count': '🧑 Young Adults (18-35)',
            'middle_aged_count': '👨 Middle-aged (36-55)',
            'seniors_count': '👴 Seniors (56+)',
            'unknown_age_count': '❓ Unknown'
        }
        total_age_persons = sum([demo[key] for key in age_labels.keys()])
        if total_age_persons > 0:
            for key, label in age_labels.items():
                count = demo[key]
                if count > 0:
                    percentage = (count / total_age_persons) * 100
                    print(f"   {label}: {count} ({percentage:.1f}%)")
    else:
        # Fallback for old format
        gender_dist = summary.get('gender_distribution', {})
        total_gender = sum(gender_dist.values())
        if total_gender > 0:
            print("👫 Gender Distribution (Frame-based - may double count):")
            for gender, count in gender_dist.items():
                if count > 0:
                    percentage = (count / total_gender) * 100
                    print(f"   {gender.capitalize()}: {count} ({percentage:.1f}%)")

        # Age distribution
        age_dist = summary.get('age_distribution', {})
        total_age = sum(age_dist.values())
        if total_age > 0:
            print("\n🎂 Age Group Distribution (Frame-based - may double count):")
            age_labels = {
                'child': '👶 Children (0-17)',
                'young': '🧑 Young Adults (18-35)',
                'middle': '👨 Middle-aged (36-55)',
                'senior': '👴 Seniors (56+)',
                'unknown': '❓ Unknown'
            }
            for age_group, count in age_dist.items():
                if count > 0:
                    percentage = (count / total_age) * 100
                    label = age_labels.get(age_group, age_group.capitalize())
                    print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Age statistics
    if summary['average_age'] > 0:
        print(f"\n📈 Age Statistics:")
        print(f"   Average Age: {summary['average_age']:.1f} years")
        if summary['age_range']['min'] and summary['age_range']['max']:
            print(f"   Age Range: {summary['age_range']['min']:.1f} - {summary['age_range']['max']:.1f} years")

def print_recent_frames(session_data: Dict, num_frames: int = 5):
    """Print details of recent frames"""
    frames = session_data['frames']
    if not frames:
        print("\n❌ No frames recorded yet.")
        return
    
    print(f"\n" + "=" * 25 + f" RECENT {min(num_frames, len(frames))} FRAMES " + "=" * 25)
    
    recent_frames = frames[-num_frames:] if len(frames) > num_frames else frames
    
    for i, frame in enumerate(recent_frames, 1):
        print(f"\n🎬 Frame {frame['frame_number']} ({frame['timestamp']})")
        print(f"   👥 Persons: {frame['total_persons']} | 😊 Faces: {frame['total_faces']}")
        
        if frame['person_detections']:
            print("   📋 Detections:")
            for j, person in enumerate(frame['person_detections'], 1):
                age = person.get('age', 'Unknown')
                gender = person.get('gender', 'Unknown')
                confidence = person.get('confidence', 0) * 100
                gender_conf = person.get('gender_confidence', 0) * 100 if person.get('gender_confidence') else 0
                
                print(f"      Person {j}: Age {age:.1f}, {gender.capitalize()} ({gender_conf:.0f}%), Conf: {confidence:.0f}%")

def list_available_sessions():
    """List all available session files"""
    analytics_dir = "analytics_data"
    if not os.path.exists(analytics_dir):
        print("❌ No analytics data directory found. Run the server first to generate data.")
        return []
    
    session_files = [f for f in os.listdir(analytics_dir) if f.startswith('session_') and f.endswith('.json')]
    
    if not session_files:
        print("❌ No session files found. Run the server and client to generate data.")
        return []
    
    print("📁 Available Sessions:")
    for i, filename in enumerate(session_files, 1):
        session_id = filename.replace('session_', '').replace('.json', '')
        filepath = os.path.join(analytics_dir, filename)
        
        # Get file modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        print(f"   {i}. {session_id} (Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return session_files

def main():
    """Main function"""
    print("🎯 MiVOLO Analytics Data Viewer")
    print("=" * 40)
    
    # List available sessions
    session_files = list_available_sessions()
    if not session_files:
        return
    
    # If command line argument provided, use it
    if len(sys.argv) > 1:
        session_choice = sys.argv[1]
        if session_choice.isdigit():
            choice_idx = int(session_choice) - 1
            if 0 <= choice_idx < len(session_files):
                selected_file = session_files[choice_idx]
            else:
                print(f"❌ Invalid choice: {session_choice}")
                return
        else:
            # Assume it's a session ID or filename
            if not session_choice.endswith('.json'):
                session_choice = f"session_{session_choice}.json"
            if session_choice in session_files:
                selected_file = session_choice
            else:
                print(f"❌ Session file not found: {session_choice}")
                return
    else:
        # Interactive selection
        try:
            choice = input(f"\n🔢 Select session (1-{len(session_files)}) or press Enter for latest: ").strip()
            if not choice:
                selected_file = session_files[-1]  # Latest file
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(session_files):
                    selected_file = session_files[choice_idx]
                else:
                    print(f"❌ Invalid choice: {choice}")
                    return
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            return
    
    # Load and display session data
    session_file_path = os.path.join("analytics_data", selected_file)
    session_data = load_session_data(session_file_path)
    
    if session_data:
        print_session_summary(session_data)
        print_recent_frames(session_data)
        
        print(f"\n💾 Full data available in: {session_file_path}")
        print("🔄 Run this script again to see updated data!")
    else:
        print("❌ Failed to load session data.")

if __name__ == "__main__":
    main()
