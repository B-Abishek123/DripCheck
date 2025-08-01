'''
Get links on the products(dress, accessories) that could improve their outfit and show the links. 

Generate a imagination(using ai maybe in ghibli style) of how the user looks after outfit upgrade. 

Space to enter user's dress in the wardrobe and pick best combination and suggest improvements from clothes available.
'''

import cv2
import time
import base64
import numpy as np
from groq import Groq
import os

def ai_model1(frame, user_data=None, occasion="casual"):

        # Handle base64 string or data URL
        if isinstance(frame, str):
            # If it's a data URL, strip the header
            if frame.startswith("data:image"):
                frame = frame.split(",", 1)[1]
            # Decode base64 to bytes
            img_bytes = base64.b64decode(frame)
            # Bytes to numpy array
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            # Decode to OpenCV image
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image from base64 string.")

        # Now frame is a numpy array, safe to encode again for LLM
        _, buffer = cv2.imencode('.jpg', frame)
        
        base64_image = base64.b64encode(buffer).decode('utf-8')


        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        chat_completion1 = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 'Find what type of top[collar tshirt, full sleeve shirt, etc.,], bottom[cargo pants, shorts, jeans, etc.,] and accessory[watch, belt, shoes, sandals, etc.,] the person is wearing in the image. '\
                                                'Also mention the color of each item. The output should be a like a python dictionary with all the data. For example:{"top":"blue sleeveless t-shirt", "bottom":"black cargo pants", "accessory":"brown leather watch"}. '\
                                                'Don"t give any other text in the output. Just give the dictionary. Nothing else. Give much details about the dress.',
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
        )


        chat_completion2 = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": str(user_data)+'These are my details. This is the occasion I am preparing for: '+str(occasion)+'.'\
                        'Now I will give you the details of my outfit right now and you please assess it according to my body, skin and provide me suggestion to look better. '\
                        'The input will be in this format: {"top":"blue sleeveless t-shirt", "bottom":"black cargo pants", "accessory":"black cateye sunglasses"}.'\
                        'You can change the type of cloth, colour of the dress or add or remove any accessories. Make the outfit a little comfortable. Be creative. Keep in mind about my appearance and reply accordingly.'\
                        'The output format should be in minimum of 3 steps(points) and maximum of 7 steps(points). Be creative and remember my appearance. The output must contain points to improve my outfit.'\
                        'The output format: "<Suggestions>". No unwanted words. I need the suggestions from second line only. Dont include header or title as "Suggestions:" Just provide the suggestions from the second line.'
                        'Please mention if there are not much improvements to be made in any part(top, bottom, accessory). Also remember the occasion I am preparing for it is also very important'
                        'Always Appreciate the user for good outfit choices and wherever possible.'
                },
                {
                        "role": "user",
                        "content": [
                        {"type": "text", "text": 'Here\'s my current outfit: '+str(chat_completion1.choices[0].message.content)},  
                    ],
                }
            ],
            model="llama-3.3-70b-versatile",
        )


        chat_completion3 = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": str(user_data)+'These are my details. This is the occasion I am preparing for: '+str(occasion)+'.You are an expert in Fashion Designing and you rank outfits based on users body details and occasion. '
                    +'Rank the current outfit of the user in scale of 1-10. The expected output is just "Rating: <rating>". Thats it. Dont include any other text or explanation. Be honest with the ratings.'\
                },
                {
                        "role": "user",
                        "content": [
                        {"type": "text", "text": 'Here\'s my current outfit: '+str(chat_completion1.choices[0].message.content)},  
                    ],
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        

        final_res=chat_completion2.choices[0].message.content.replace("\n", "\n\n")
        return [final_res, chat_completion3.choices[0].message.content]










def main2_manual_bodydetails(age, gender, weight, height, body_type, skin_tone, hair_color, face_shape, liked_colors, hated_colors, other_preferences, occasion):
                data = f"""I am a {age} year old {gender} weighing {weight}kg. I am {height}cms tall. I am {body_type}.I have {skin_tone} skintone with {hair_color} hair.
                I have a {face_shape} face. I like {liked_colors} colors and I hate {hated_colors}. 
                My other preferences: {other_preferences}. This is the occasion I am preparing for: {occasion}. """
                time.sleep(1.5)
                return data





def main1_image_bodydetails(image_path):
    img_stream = image_path.read()  # `img` is a FileStorage object (from `request.files`)
    base64_image = base64.b64encode(img_stream).decode('utf-8')


    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 'Find these details about the person in the image. Details to be found:'
                    '1. Age\n2. Race\n3. Gender\n4. Approximate Weight\n5. Approximate Height\n6. Body Type\n7. Skin Tone\n8. Hair Color\n9. Face Shape\n10. Facial Hair Style\n 11. Additional information about the person\'s body or face.\n'
                    'If any of the details is not found, no need to mention it. The output should be only the details of the person. '\
                    'Do not use special characters in the output. If the image is not clear, please ask to retake the photo. Analyse properly. Don\'t add information about the dress or accessories that the person is wearing.'
                    'Don\'t give any details about other objects or person in the surroundings. If there are more than 2 persons, take the details of person who takes more space in the frame.'
                    'Example Output: "The person is around 30 years of age, female gender, approximately 70kg, 180cm. She has a medium body type with white skin, grey hair color, oval shape and no facial hair. She has a short hair cut, wears earrings"'\
                    'Analyse details of the person\'s body type and face and skin. Don\'t add unnecessary information.'\
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
    )
    return chat_completion.choices[0].message.content





def main1_capture_bodydetails():
    cap = cv2.VideoCapture(0)
    # Get frame dimensions for positioning text in top-right corner
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]


    # Countdown phase (6 seconds)
    countdown_start = time.time()
    countdown_duration = 5
    while True:
        ret, frame = cap.read()
        
        # Calculate remaining time
        elapsed = time.time() - countdown_start
        remaining = countdown_duration - elapsed
        
        if remaining <= 0:
            
            # Display "CAPTURED!" message
            captured_frame = frame.copy()
            
            # Add "CAPTURED!" text in center
            text = "CAPTURED!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            color = (0, 255, 0)  # Green color
            thickness = 3
            
            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (frame_width - text_width) // 2
            text_y = (frame_height + text_height) // 2
            
            cv2.putText(captured_frame, text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Show captured frame for 2 seconds
            display_start = time.time()
            while time.time() - display_start < 2:
                cv2.imshow("Webcam", captured_frame)
                cv2.waitKey(30)
            break
        
        # Display countdown in top-right corner
        countdown_text = str(int(remaining) + 1)  # Show 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4  # Increased from 2 to 4 (2x bigger)
        color = (0, 0, 0)  # Red color for countdown
        thickness = 6  # Increased from 3 to 6 (2x bigger)
        
        # Calculate position for top-right corner
        (text_width, text_height), baseline = cv2.getTextSize(countdown_text, font, font_scale, thickness)
        text_x = frame_width - text_width - 20  # 20 pixels from right edge
        text_y = text_height + 20  # 20 pixels from top edge
        
        # Add background rectangle for better visibility (reduced length by 5px)
        cv2.rectangle(frame,
                    (text_x - 5, text_y - text_height - 10),  # Reduced left padding from 10 to 5
                    (text_x + text_width + 5, text_y + baseline - 5),  # Reduced right padding from 10 to 5
                    (255, 255, 255), -1) 
        cv2.putText(frame, countdown_text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Show the frame
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(30) & 0xFF

    cap.release()
    cv2.destroyAllWindows()
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')



    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 'Find these details about the person in the image. Details to be found:'
                    '1. Age\n2. Race\n3. Gender\n4. Approximate Weight\n5. Approximate Height\n6. Body Type\n7. Skin Tone\n8. Hair Color\n9. Face Shape\n10. Facial Hair Style\n 11. Additional information about the person\'s body or face.\n'
                    'If any of the details is not found, no need to mention it. The output should be only the details of the person. '\
                    'Do not use special characters in the output. If the image is not clear, please ask to retake the photo. Analyse properly. Don\'t add information about the dress or accessories that the person is wearing.'
                    'Don\'t give any details about other objects or person in the surroundings. If there are more than 2 persons, take the details of person who takes more space in the frame.'
                    'Example Output: "The person is around 30 years of age, female gender, approximately 70kg, 180cm. She has a medium body type with white skin, grey hair color, oval shape and no facial hair. She has a short hair cut, wears earrings"'\
                    'Analyse details of the person\'s body type and face and skin. Don\'t add unnecessary information.'\
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
    )

    return chat_completion.choices[0].message.content
        
