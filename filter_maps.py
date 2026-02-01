import os
import subprocess
import shutil

def run_evaluation():
    # Define paths
    source_folder = "gen_maps"
    destination_folder = "selected"
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Iterate through all files in gen_maps
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(source_folder, filename)
            
            # Construct the terminal command
            command = [
                "python", "src/game.py",
                "--red", "bots/ethan_bot.py",
                "--blue", "bots/default_bot.py",
                "--map", file_path
            ]
            
            try:
                # Run command and capture output
                result = subprocess.run(command, capture_output=True, text=True)
                output = result.stdout
                
                # Look for the score line in the output
                # Example: [GAME OVER] money scores: RED=1000, BLUE=2500
                if "[GAME OVER]" in output:
                    # Find the part with the scores
                    # Split by RED= and BLUE= to extract numbers
                    try:
                        # Extract the section after 'RED='
                        red_part = output.split("RED=$")[1].split(",")[0]
                        # Extract the section after 'BLUE='
                        blue_part = output.split("BLUE=$")[1].split("\n")[0].strip()
                        
                        r_score = int(red_part)
                        b_score = int(blue_part)
                        
                        # Check the condition: b_score > r_score + 1000
                        if r_score > b_score + 1000:
                            print(f"Match found! Copying {filename} (R:{r_score}, B:{b_score})")
                            shutil.copy(file_path, os.path.join(destination_folder, filename))
                        else:
                            print(f"Skipping {filename}: Scores R:{r_score}, B:{b_score}")
                            
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing scores in {filename}: {e}")
                else:
                    print(f"Game did not finish properly for {filename}")
                    
            except Exception as e:
                print(f"Failed to run game for {filename}: {e}")

if __name__ == "__main__":
    run_evaluation()