# Django Models for Camera Analytics
# models.py

from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid

class CameraSession(models.Model):
    """Represents a camera session/stream"""
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    camera_name = models.CharField(max_length=100, help_text="Name/identifier of the camera")
    location = models.CharField(max_length=200, help_text="Physical location of the camera")
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        app_label = 'analytics'
        db_table = 'camera_sessions'
        ordering = ['-start_time']
    
    def __str__(self):
        return f"{self.camera_name} - {self.start_time.strftime('%Y-%m-%d %H:%M')}"

class AnalyticsFrame(models.Model):
    """Represents analytics data for a single frame"""
    frame_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(CameraSession, on_delete=models.CASCADE, related_name='frames')
    timestamp = models.DateTimeField(default=timezone.now)
    frame_number = models.PositiveIntegerField()
    
    # Overall counts
    total_persons = models.PositiveIntegerField(default=0)
    total_faces = models.PositiveIntegerField(default=0)
    
    # Gender counts
    male_count = models.PositiveIntegerField(default=0)
    female_count = models.PositiveIntegerField(default=0)
    unknown_gender_count = models.PositiveIntegerField(default=0)
    
    # Age group counts
    children_count = models.PositiveIntegerField(default=0, help_text="Age 0-17")
    young_adults_count = models.PositiveIntegerField(default=0, help_text="Age 18-35")
    middle_aged_count = models.PositiveIntegerField(default=0, help_text="Age 36-55")
    seniors_count = models.PositiveIntegerField(default=0, help_text="Age 56+")
    unknown_age_count = models.PositiveIntegerField(default=0)
    
    # Processing metadata
    processing_time_ms = models.FloatField(help_text="Time taken to process this frame in milliseconds")
    confidence_threshold = models.FloatField(default=0.5)
    
    class Meta:
        app_label = 'analytics'
        db_table = 'analytics_frames'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['session', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]
    
    def __str__(self):
        return f"Frame {self.frame_number} - {self.total_persons} persons"

class PersonDetection(models.Model):
    """Individual person detection within a frame"""
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('U', 'Unknown'),
    ]
    
    AGE_GROUP_CHOICES = [
        ('CHILD', 'Child (0-17)'),
        ('YOUNG', 'Young Adult (18-35)'),
        ('MIDDLE', 'Middle Aged (36-55)'),
        ('SENIOR', 'Senior (56+)'),
        ('UNKNOWN', 'Unknown'),
    ]
    
    detection_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    frame = models.ForeignKey(AnalyticsFrame, on_delete=models.CASCADE, related_name='person_detections')
    
    # Bounding box coordinates (normalized 0-1)
    bbox_x = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    bbox_y = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    bbox_width = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    bbox_height = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    # Person attributes
    predicted_age = models.PositiveIntegerField(null=True, blank=True)
    age_group = models.CharField(max_length=10, choices=AGE_GROUP_CHOICES, default='UNKNOWN')
    predicted_gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='U')
    
    # Confidence scores
    person_confidence = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    age_confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    gender_confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    # Tracking information
    track_id = models.PositiveIntegerField(null=True, blank=True, help_text="YOLO tracking ID")
    
    class Meta:
        app_label = 'analytics'
        db_table = 'person_detections'
        indexes = [
            models.Index(fields=['frame', 'track_id']),
            models.Index(fields=['predicted_gender']),
            models.Index(fields=['age_group']),
        ]
    
    def save(self, *args, **kwargs):
        # Auto-determine age group based on predicted age
        if self.predicted_age is not None:
            if self.predicted_age <= 17:
                self.age_group = 'CHILD'
            elif self.predicted_age <= 35:
                self.age_group = 'YOUNG'
            elif self.predicted_age <= 55:
                self.age_group = 'MIDDLE'
            else:
                self.age_group = 'SENIOR'
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Person {self.detection_id} - {self.get_predicted_gender_display()}, Age: {self.predicted_age or 'Unknown'}"

class FaceDetection(models.Model):
    """Individual face detection within a frame"""
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('U', 'Unknown'),
    ]
    
    detection_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    frame = models.ForeignKey(AnalyticsFrame, on_delete=models.CASCADE, related_name='face_detections')
    person_detection = models.ForeignKey(PersonDetection, on_delete=models.CASCADE, null=True, blank=True, 
                                       related_name='associated_faces', help_text="Associated person detection if available")
    
    # Bounding box coordinates (normalized 0-1)
    bbox_x = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    bbox_y = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    bbox_width = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    bbox_height = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    # Face attributes
    predicted_age = models.PositiveIntegerField(null=True, blank=True)
    predicted_gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='U')
    
    # Confidence scores
    face_confidence = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    age_confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    gender_confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    class Meta:
        app_label = 'analytics'
        db_table = 'face_detections'
        indexes = [
            models.Index(fields=['frame']),
            models.Index(fields=['predicted_gender']),
        ]
    
    def __str__(self):
        return f"Face {self.detection_id} - {self.get_predicted_gender_display()}, Age: {self.predicted_age or 'Unknown'}"

class SessionSummary(models.Model):
    """Aggregated analytics summary for a camera session"""
    session = models.OneToOneField(CameraSession, on_delete=models.CASCADE, related_name='summary')
    
    # Total counts
    total_frames_processed = models.PositiveIntegerField(default=0)
    unique_persons_detected = models.PositiveIntegerField(default=0)
    total_person_detections = models.PositiveIntegerField(default=0)
    total_face_detections = models.PositiveIntegerField(default=0)
    
    # Gender distribution
    total_males = models.PositiveIntegerField(default=0)
    total_females = models.PositiveIntegerField(default=0)
    total_unknown_gender = models.PositiveIntegerField(default=0)
    
    # Age distribution
    total_children = models.PositiveIntegerField(default=0)
    total_young_adults = models.PositiveIntegerField(default=0)
    total_middle_aged = models.PositiveIntegerField(default=0)
    total_seniors = models.PositiveIntegerField(default=0)
    total_unknown_age = models.PositiveIntegerField(default=0)
    
    # Performance metrics
    avg_processing_time_ms = models.FloatField(default=0.0)
    max_persons_in_frame = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        app_label = 'analytics'
        db_table = 'session_summaries'
    
    def __str__(self):
        return f"Summary for {self.session.camera_name} - {self.unique_persons_detected} unique persons"
