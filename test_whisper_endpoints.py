#!/usr/bin/env python3
"""
Test Whisper Server Endpoints

This script tests various Whisper server endpoints to verify functionality.
Based on the patterns seen in the markdown files, this tests:
1. Health endpoint
2. V2T endpoint
3. Transcribe endpoint
4. Error handling
"""

import requests
import json
import numpy as np
import base64
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhisperEndpointTester:
    """Test Whisper server endpoints."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = requests.Session()
        self.session.timeout = 60
        
    def create_test_audio(self, duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
        """Create test audio signal."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Generate a simple sine wave
        frequency = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        return audio.astype(np.float32)
    
    def audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Convert audio array to base64 string."""
        audio_bytes = audio.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_b64
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health endpoint."""
        logger.info("🔍 Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.server_url}/health")
            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
            if result["success"]:
                logger.info(f"✅ Health check: {result['status_code']}")
                logger.info(f"📊 Response: {result['response']}")
            else:
                logger.error(f"❌ Health check failed: {result['status_code']}")
                logger.error(f"Response: {result['response']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Health endpoint test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_v2t_endpoint(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Test the v2t endpoint."""
        logger.info("🎤 Testing v2t endpoint...")
        
        try:
            audio_b64 = self.audio_to_base64(audio, sample_rate)
            request_data = {
                "audio_data": audio_b64,
                "sample_rate": sample_rate
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.server_url}/api/v1/v2t", json=request_data)
            response_time = time.time() - start_time
            
            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response_time,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
            if result["success"]:
                transcription = result["response"].get("text", "")
                logger.info(f"✅ V2T response: {response.status_code}")
                logger.info(f"⏱️  Response time: {response_time:.3f}s")
                logger.info(f"📝 Transcription: {transcription}")
            else:
                logger.error(f"❌ V2T failed: {result['status_code']}")
                logger.error(f"Response: {result['response']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ V2T endpoint test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_transcribe_endpoint(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Test the transcribe endpoint."""
        logger.info("🎤 Testing transcribe endpoint...")
        
        try:
            audio_b64 = self.audio_to_base64(audio, sample_rate)
            request_data = {
                "audio_data": audio_b64,
                "sample_rate": sample_rate
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.server_url}/api/v1/transcribe", json=request_data)
            response_time = time.time() - start_time
            
            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response_time,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
            if result["success"]:
                transcription = result["response"].get("text", "")
                logger.info(f"✅ Transcribe response: {response.status_code}")
                logger.info(f"⏱️  Response time: {response_time:.3f}s")
                logger.info(f"📝 Transcription: {transcription}")
            else:
                logger.error(f"❌ Transcribe failed: {result['status_code']}")
                logger.error(f"Response: {result['response']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Transcribe endpoint test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid requests."""
        logger.info("🔍 Testing error handling...")
        
        test_cases = [
            {
                "name": "Invalid JSON",
                "data": "invalid json",
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "Missing audio_data",
                "data": {"sample_rate": 16000},
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "Invalid sample_rate",
                "data": {"audio_data": "test", "sample_rate": "invalid"},
                "headers": {"Content-Type": "application/json"}
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.server_url}/api/v1/v2t",
                    data=test_case["data"] if isinstance(test_case["data"], str) else json.dumps(test_case["data"]),
                    headers=test_case["headers"]
                )
                
                result = {
                    "name": test_case["name"],
                    "success": response.status_code >= 400,  # Expecting error status
                    "status_code": response.status_code,
                    "response": response.text
                }
                
                if result["success"]:
                    logger.info(f"✅ {test_case['name']}: {result['status_code']} (expected error)")
                else:
                    logger.warning(f"⚠️  {test_case['name']}: {result['status_code']} (unexpected success)")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ {test_case['name']} test failed: {e}")
                results.append({
                    "name": test_case["name"],
                    "success": False,
                    "error": str(e)
                })
        
        return {"test_cases": results}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive endpoint tests."""
        logger.info("🚀 Testing Whisper Server Endpoints")
        logger.info("=" * 50)
        
        # Create test audio
        audio = self.create_test_audio()
        sample_rate = 16000
        
        # Run tests
        health_result = self.test_health_endpoint()
        v2t_result = self.test_v2t_endpoint(audio, sample_rate)
        transcribe_result = self.test_transcribe_endpoint(audio, sample_rate)
        error_result = self.test_error_handling()
        
        # Compile results
        results = {
            "health": health_result,
            "v2t": v2t_result,
            "transcribe": transcribe_result,
            "error_handling": error_result,
            "overall_success": (
                health_result.get("success", False) and
                v2t_result.get("success", False) and
                transcribe_result.get("success", False)
            )
        }
        
        # Print summary
        logger.info("\n📊 Test Summary:")
        logger.info(f"Health Endpoint: {'✅' if health_result.get('success') else '❌'}")
        logger.info(f"V2T Endpoint: {'✅' if v2t_result.get('success') else '❌'}")
        logger.info(f"Transcribe Endpoint: {'✅' if transcribe_result.get('success') else '❌'}")
        logger.info(f"Error Handling: {'✅' if error_result.get('test_cases') else '❌'}")
        
        if results["overall_success"]:
            logger.info("\n✅ All endpoint tests passed!")
        else:
            logger.error("\n❌ Some endpoint tests failed!")
        
        return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Whisper Server Endpoints")
    parser.add_argument("--server-url", default="http://localhost:8000",
                       help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--output-file", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    tester = WhisperEndpointTester(args.server_url)
    results = tester.run_comprehensive_test()
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📁 Results saved to: {args.output_file}")
    
    exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    main()





