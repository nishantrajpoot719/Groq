import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from pydantic import BaseModel
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize Gradio client
GRADIO_CLIENT = Client("nishantrajpoot/must_duplicate")


def setup_logging() -> logging.Logger:
    """Set up and configure the logger.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Prevent adding multiple handlers if function is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
logger = setup_logging()


@app.get("/api/process_video")
async def process_video_endpoint(video_input: str):           
        gradio_result = GRADIO_CLIENT.predict(
            video_input={"video": handle_file(video_input)},
            api_name="/process_video"
        )

        # Extract VAD scores 
        vad_score = gradio_result.get('Final VAD Score', [0, 0, 0]) 
        # Get contextual data 
        contextual_data = gradio_result.get('Contextual Information', [])  

        analysis_data = {
            'vad_score': vad_score,
            'contextual_data': contextual_data
        }
        
        # Get food recommendations using only VAD scores and contextual data
        food_recommendations = get_food_recommendations(analysis_data)
        
        return JSONResponse(content={
            "data": food_recommendations,
        })


def get_food_recommendations(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the analysis result and return food recommendations using Groq API.
    
    Args:
        analysis_result: Dictionary containing vad_score, and contextual_data
        
    Returns:
        Dict containing food recommendations and analysis
    """
    try:
        # Get VAD scores (Valence, Arousal, Dominance)
        vad_score = analysis_result.get('vad_score', [0, 0, 0])
            
        # Get contextual data
        contextual_data = analysis_result.get('contextual_data', [])
        
        # Default intent (can be overridden by contextual data)
        intent = ["Hot", "Light", "Tangy"]
            
        # Prepare input for Groq API
        input_data = [vad_score, intent, contextual_data]
        
        # Initialize Groq client
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

        try:
            client = Groq(api_key=GROQ_API_KEY)
            # Call the Groq API
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """This system recommends top three products and top three product combos based on user emotion (VAD score), preferences (Intent), and contextual data (time, date, location, weather).

Input: [VAD, Intent, Contextual Data]
- VAD: [Valence, Arousal, Dominance] values from -1 to 1, indicating user emotion. Used for initial product decision.
- Intent: User preferences (e.g., ["Hot", "Light", "Tangy"]). If incomplete, consider only provided preferences.
- Contextual Data: [time, date, location, weather]. Used for rearranging recommendations based on suitability (e.g., time of day, festivals, weather). Contextual data enhances recommendations but does not alter product decisions.

Product Dictionary: (Assumed to be accessible to the model through its training or external context, not embedded in prompt)
- Recommend only one variant per product.
- Products with inherent combos (e.g., Muesli with milk, Malabar Parata with pickles) should be recommended as combos.
- Chips and puffs can be bundled as single products.
- For products with variants, use the variant name in the output (e.g., "Roasted Makhana" instead of "Makhana - Roasted Makhana").
- Combos should be based on explicit product dictionary information, not random combinations.

Output must be in JSON format and include:
1. Emotion of the user from the VAD score (one or two self-explanatory words).
2. Top three products recommended as a list.
3. Top three combos recommended as a list."""
                    },
                    {
                        "role": "user",
                        "content": str(input_data)
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                response_format={"type": "json_object"}
            )
            
            # Process the response
            response_content = completion.choices[0].message.content
            
            # Parse the JSON response
            recommendations = json.loads(response_content)
            
            # Normalize the response format
            normalized_recommendations = {}
            for key, value in recommendations.items():
                if key.lower() == 'emotion':
                    normalized_key = 'emotion'
                elif key.lower() == 'top products':
                    normalized_key = 'top_products'
                elif key.lower() == 'top combos':
                    normalized_key = 'top_combos'
                else:
                    normalized_key = key
                normalized_recommendations[normalized_key] = value
            
            # Ensure emotion is a string
            if "emotion" in normalized_recommendations:
                if isinstance(normalized_recommendations["emotion"], list):
                    normalized_recommendations["emotion"] = ' '.join([str(word).strip() for word in normalized_recommendations["emotion"]])
                elif isinstance(normalized_recommendations["emotion"], str):
                    normalized_recommendations["emotion"] = normalized_recommendations["emotion"].strip()
            
            # Format the final response
            response_data = {
                "emotion": normalized_recommendations.get("emotion", ""),
                "top_products": normalized_recommendations.get("top_products", []),
                "top_combos": normalized_recommendations.get("top_combos", []),
                "message": "Recommendations generated successfully"
            }
            
            return response_data
            
        except Exception as api_error:
            logger.error(f"Error calling Groq API: {str(api_error)}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Failed to get recommendations from the AI service"
            )
            
    except json.JSONDecodeError as je:
        logger.error(f"Error decoding API response: {str(je)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse API response"
        )
        
        
    except Exception as e:
        logger.error(f"Unexpected error in get_food_recommendations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

