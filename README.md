# SEIVR++

**SEIVR++ (Semantic Event Indexed Video Retrieval System)** is an AI-powered video intelligence system that transforms surveillance footage into structured semantic events. Instead of manually reviewing long videos, users can retrieve specific events using natural language queries.

## Features

- Object detection using YOLO
- Multi-object tracking with ByteTrack
- Action recognition using Vision Transformer (ViT)
- Spatial zone reasoning
- Semantic event generation
- Structured JSON event storage
- Natural language event retrieval

## Tech Stack

- Python
- PyTorch
- OpenCV
- Ultralytics YOLO
- ByteTrack
- Vision Transformer (ViT)
- Microsoft Phi-3 Mini / Groq (Query Parsing)

## Workflow

```
Video
   ↓
Object Detection
   ↓
Object Tracking
   ↓
Action Recognition
   ↓
Spatial Reasoning
   ↓
Event Generation
   ↓
Event Storage
   ↓
Natural Language Query
```

## Current Capabilities

- Person detection
- Vehicle detection
- Running, walking, and standing recognition
- Umbrella detection
- Color identification
- Zone-based reasoning
- Semantic event storage and retrieval

## Future Improvements

- Temporal action recognition using TimeSformer/ViViT
- Advanced trajectory and lane transition analysis
- Semantic vector search (RAG)
- Real-time CCTV stream support
- Multi-camera event fusion

## Project Goal

To build an intelligent surveillance system capable of understanding video content, generating semantic events, and enabling efficient natural language retrieval of relevant activities.
