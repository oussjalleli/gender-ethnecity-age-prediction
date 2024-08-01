import os
import pandas as pd
from deepface import DeepFace

def analyze_images(folder_path, output_csv_path):
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg') or file.endswith('.png')]

    results_df = pd.DataFrame(columns=["#", "filename", "age", "gender", "race"])
    image_count = 0  # Counter for the number of images

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image_count += 1  # Increment the counter

        try:
            results = DeepFace.analyze(image_path, actions=['age', 'gender', 'race'], enforce_detection=False)

            # Check if results is empty (no face detected)
            if not results:
                row = pd.DataFrame([{
                    "#": image_count,
                    "filename": image_file,
                    "age": -1,
                    "gender": "unknown",
                    "race": "unknown"
                }])
                results_df = pd.concat([results_df, row], ignore_index=True)
            else:
                for face_result in results:
                    row = pd.DataFrame([{
                        "#": image_count,
                        "filename": image_file,
                        "age": face_result['age'],
                        "gender": face_result['dominant_gender'],
                        "race": face_result['dominant_race']
                    }])
                    results_df = pd.concat([results_df, row], ignore_index=True)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

folder_path = 'facepics'
output_csv_path = 'face_analysis_results.csv'
analyze_images(folder_path, output_csv_path)