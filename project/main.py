import os
from uno_game_state_extractor import UnoGameStateExtractor

def evaluate_first_image():
    # Define your specific folder paths
    training_dir = "training"
    template_dir = "templates"

    # 1. Verify directories exist
    if not os.path.exists(training_dir):
        print(f"Error: Could not find training directory '{training_dir}'")
        return
        
    # 2. Get all images and sort them to reliably grab the first one
    image_files = [f for f in os.listdir(training_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Error: No images found in '{training_dir}'")
        return
        
    image_files.sort()
    first_image_filename = image_files[1]
    first_image_path = os.path.join(training_dir, first_image_filename)
    
    # Extract the ID (filename without extension, e.g., "L1000777")
    image_id = os.path.splitext(first_image_filename)[0] 

    print(f"--- Starting Evaluation ---")
    print(f"Image File: {first_image_path}")
    print(f"Image ID:   {image_id}")
    print(f"---------------------------\n")

    # 3. Instantiate your pipeline
    extractor = UnoGameStateExtractor(template_dir=template_dir)

    # 4. Process the image
    try:
        result_row = extractor.process_image(first_image_path, image_id)
        
        # Print the exact CSV string format
        csv_string = ",".join(result_row)
        print("Raw CSV Output:")
        print(csv_string)
        print("\n---------------------------")
        
        # Print a human-readable breakdown to verify your logic
        print("Breakdown:")
        print(f"Background:    {extractor.last_bg_type.upper()}") 
        print(f"Center Card:   {result_row[1]}")
        print(f"Active Player: {result_row[2]}")
        print(f"Player 1 (Bottom): {result_row[3]}")
        print(f"Player 2 (Right):  {result_row[4]}")
        print(f"Player 3 (Top):    {result_row[5]}")
        print(f"Player 4 (Left):   {result_row[6]}")

    except Exception as e:
        print(f"Pipeline crashed during execution: {e}")

if __name__ == "__main__":
    evaluate_first_image()