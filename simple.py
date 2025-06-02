from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)
client = Client("nishantrajpoot/must_duplicate")

@app.route("/", methods=["GET"])
def running():
    return "Welcome to the Must Duplicate API!"

@app.route("/process_video", methods=["POST","GET"])
def process_video():
    data = request.args
    video_url = data.get("video_url")
    result = client.predict(
        video_input={"video": handle_file(video_url)},
        api_name="/process_video"
    )
    # Extract VAD scores 
    vad_score = result.get('Final VAD Score', [0, 0, 0]) 
    # Get contextual data 
    contextual_data = result.get('Contextual Information', [])  

    analysis_data = {
            'vad_score': vad_score,
            'contextual_data': contextual_data
        }
    
    food_recommendations = get_food_recommendations(analysis_data)
    


def get_food_recommendations(analysis_result):
    vad_score = analysis_result.get('vad_score')
    contextual_data = analysis_result.get('contextual_data')
    intent = ["Hot", "Light", "Tangy"]
    input_data = [vad_score, intent, contextual_data]

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)

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

    response_content = completion.choices[0].message.content
    recommendations = json.loads(response_content)

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

    if "emotion" in normalized_recommendations:
        if isinstance(normalized_recommendations["emotion"], list):
            normalized_recommendations["emotion"] = ' '.join([str(word).strip() for word in normalized_recommendations["emotion"]])
        elif isinstance(normalized_recommendations["emotion"], str):
            normalized_recommendations["emotion"] = normalized_recommendations["emotion"].strip()

    response_data = {
        "emotion": normalized_recommendations.get("emotion", ""),
        "top_products": normalized_recommendations.get("top_products", []),
        "top_combos": normalized_recommendations.get("top_combos", []),
        "message": "Recommendations generated successfully"
    }
            
    return jsonify(response_data)

if _name_ == "_main_":
    app.run(debug = True)
