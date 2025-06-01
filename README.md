---
title: Food Recommendation API
emoji: 🍽️
colorFrom: green
colorTo: purple
sdk: gradio
python_version: 3.10.11
sdk_version: 5.31.0
app_file: huggingface_app.py
pinned: false
---

# Food Recommendation API

A Flask-based API that provides personalized food recommendations based on emotional state (VAD scores), user preferences, and contextual data. The system integrates with the Groq API for generating intelligent recommendations.

## Features

- **Emotion-Based Recommendations**: Uses Valence-Arousal-Dominance (VAD) scores to suggest appropriate foods
- **Context-Aware**: Considers time of day, weather, and location for relevant suggestions
- **Personalized Preferences**: Takes into account user's food preferences and dietary restrictions
- **Combo Suggestions**: Recommends food combinations for a complete meal experience
- **RESTful API**: Easy integration with web and mobile applications

## API Endpoints

### 1. Health Check
- **Endpoint**: `GET /`
- **Response**: `{"status": "healthy", "message": "Food Recommendation API is running"}`

### 2. Get Food Recommendations
- **Endpoint**: `POST /recommend`
- **Request Body**:
  ```json
  {
    "vad_score": [float, float, float],
    "intent_selections": ["string"],
    "contextual_data": ["time", "date", "location", "weather"]
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "emotion": "string",
    "top_products": ["string"],
    "top_combos": ["string"],
    "message": "string"
  }
  ```

## Getting Started

### Prerequisites
- Python 3.9+
- Groq API key (get it from [Groq Console](https://console.groq.com/))

### Local Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd VAD-main
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Run the application:
   ```bash
   python food_recommendations.py
   ```

   The API will be available at `http://localhost:5000`

### Testing the API

1. Run the test script:
   ```bash
   python test_api.py
   ```

2. Or use curl:
   ```bash
   curl -X POST http://localhost:5000/recommend \
   -H "Content-Type: application/json" \
   -d '{
     "vad_score": [0.5, 0.3, 0.2],
     "intent_selections": ["Light", "Healthy"],
     "contextual_data": ["afternoon", "2025-06-01", "Mumbai", "sunny"]
   }'
   ```

## Deployment

### Render (Recommended)

1. Push your code to a Git repository
2. Create a new Web Service on Render
3. Connect your repository
4. Set the following environment variables:
   - `GROQ_API_KEY`: Your Groq API key
   - `PYTHON_VERSION`: 3.9.18
5. Set the build command: `pip install -r requirements.txt`
6. Set the start command: `gunicorn food_recommendations:app`

## Security

- Rate limiting: 200 requests per day, 50 per hour (configurable)
- Request timeout: 45 seconds
- Security headers enabled (CSP, XSS Protection, etc.)
- Input validation for all endpoints

## Error Handling

The API returns appropriate HTTP status codes and JSON error messages for various scenarios:
- `400 Bad Request`: Invalid input data
- `401 Unauthorized`: Missing or invalid API key
- `500 Internal Server Error`: Server-side error
- `504 Gateway Timeout`: Request timed out

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Gradio app:
   ```bash
   python gradio_app.py
   ```

5. Open the provided local URL (typically http://localhost:7860) in your browser

## Deployment on Hugging Face Spaces

### Prerequisites
- A Hugging Face account
- Git installed on your local machine

### Step 1: Configure Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click "Create new Space"
2. Configure your space:
   - Name: Choose a name (e.g., `vad-food-recommendation`)
   - License: MIT
   - Space SDK: Select "Docker"
   - Visibility: Public or Private as per your preference
   - Hardware: Select a GPU instance for better performance
   - Environment variables: No additional variables needed by default

### Step 2: Push Your Code

1. Initialize a Git repository (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Add the Hugging Face repository as a remote:
   ```bash
   git remote add origin https://huggingface.co/spaces/your-username/your-space-name
   ```
   Replace `your-username` and `your-space-name` with your actual Hugging Face username and space name.

3. Push your code:
   ```bash
   git push -u origin main
   ```

### Step 3: Monitor the Deployment

1. After pushing, the build process will start automatically
2. Monitor the build logs in the "Logs" tab of your Space
3. Once built, your application will be available at:
   ```
   https://huggingface.co/spaces/your-username/your-space-name
   ```

### Configuration Options

1. **Hardware Acceleration**:
   - For better performance, select a GPU instance in the Space settings
   - The application supports both CPU and GPU, but GPU is recommended for faster inference

2. **Environment Variables**:
   - The app is pre-configured with default settings
   - No additional environment variables are required for basic functionality

3. **Build Process**:
   - The Dockerfile handles all necessary installations
   - Build typically takes 5-10 minutes depending on the hardware

### Troubleshooting

1. **Build Failures**:
   - Check the logs for specific error messages
   - Ensure all required files are included in the repository
   - Verify that the Dockerfile is in the root directory

2. **Runtime Issues**:
   - Check the application logs in the Hugging Face Space
   - Ensure the selected hardware meets the requirements
   - If using GPU, verify that CUDA is properly configured

3. **Common Errors**:
   - **FFmpeg not found**: The Dockerfile includes FFmpeg installation
   - **Model loading issues**: Ensure all model files are included in the repository
   - **Memory errors**: Consider upgrading to a higher-tier GPU or optimizing model loading

### Performance Considerations

1. **Cold Start**: The first request may take longer as models are loaded into memory
2. **Concurrent Users**: For handling multiple users, consider upgrading the hardware
3. **Caching**: The application implements basic caching for better performance with repeated requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.
