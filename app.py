"""
Remove right side 2 buttons in nav bar and put the dripcheck button in the center. Update bg. Change the subline(Your personal....). Make a new section for Body Details. Add a footer.
"""



from flask import Flask, render_template, request, jsonify
import os
import base64
from random import randint
import sys
from dotenv import load_dotenv

# Add path to import stylesync
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../other python versions')))
import stylesync

load_dotenv()

app = Flask(__name__)


# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('DripCheck.html')



@app.route('/image_body_details', methods=['POST'])
def image_body_details():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        img = request.files['image']
        if img.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if allowed_file(img.filename):            
            if img is None:
                return jsonify({'error': 'Could not read image'}), 400
                
            # Process image with stylesync
            result = stylesync.main1_image_bodydetails(img)
            return jsonify({"body_details": result})
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        print(f"Error in image_body_details: {str(e)}")
        return jsonify({"error": str(e)}), 500




@app.route('/manual_body_details', methods=['POST'])
def manual_body_details():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Map frontend fields to main2 arguments
        result = stylesync.main2_manual_bodydetails(
            age=str(data.get("age", "25")),
            gender=str(data.get("gender", "Male")),
            weight=str(data.get("weight", "70")),
            height=str(data.get("height", "170")),
            body_type=str(data.get("bodyType", "Average")),
            skin_tone=str(data.get("skinTone", "Medium")),
            hair_color=str(data.get("hairColor", "Black")),
            face_shape=str(data.get("faceShape", "Oval")),
            liked_colors="None",
            hated_colors="None",
            other_preferences=str(data.get("preferences", "None")),
            occasion=str(data.get("occasion", "Casual"))
        )

        return jsonify({"body_details": result})
    except Exception as e:
        print(f"Error in body_details_manual: {str(e)}")
        return jsonify({"error": str(e)}), 500




@app.route('/recommend/image', methods=['POST'])
def recommend_image():
    try:
        img = request.files['image']
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        if img.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if img.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if img and allowed_file(img.filename):
            # Read image file
            user_data = request.form['user_data']
            occasion = request.form['occasion']
            
            if img is None:
                return jsonify({'error': 'Could not read image'}), 400
                
            # Process image with stylesync
            img_stream = img.read()  # `img` is a FileStorage object (from `request.files`)
            base64_image = base64.b64encode(img_stream).decode('utf-8')
            result = stylesync.ai_model1(base64_image, user_data, occasion)
            return jsonify({
                'status': 'success',
                'rating': result[1],
                'recommendation': result[0]
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        print(f"Error in recommend_image: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/recommend/capture', methods=['POST'])
def recommend_capture():
    try:
        # Get and validate JSON data
        data = request.get_json(silent=True)
        if not data:
            error_msg = "No JSON data received"
            print(error_msg)
            return jsonify({'error': error_msg, 'details': 'Request must be JSON'}), 400
            
        
        # Validate required fields
        required_fields = ['outfitImage', 'occasion', 'user_data']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            print(error_msg)
            return jsonify({'error': 'Invalid request', 'missing_fields': missing_fields}), 400
            
        # Extract and validate data
        try:
            outfitImage = data["outfitImage"]
            occasion = data["occasion"]
            user_data = data["user_data"]
            
            # Basic validation of the image data
            if not isinstance(outfitImage, str) or not outfitImage.startswith('data:image'):
                error_msg = "Invalid image format"
                print(error_msg)
                return jsonify({'error': error_msg, 'details': 'Invalid image data'}), 400
                
            
            # Process the request
            response = stylesync.ai_model1(outfitImage, user_data, occasion)
            
            if not response:
                error_msg = "No response generated from AI model"
                print(error_msg)
                return jsonify({'error': 'Processing failed', 'details': error_msg}), 500
                
            return jsonify({
                'status': 'success',
                'rating': response[1],
                'recommendation': response[0]
            })
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': 'Processing error',
                'details': str(e),
                'type': type(e).__name__
            }), 500
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'type': type(e).__name__
        }), 500
    




if __name__ == '__main__':
    app.run(debug=False)
